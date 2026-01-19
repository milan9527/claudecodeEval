#!/usr/bin/env python3
"""
Session Evaluation for Claude Code Offline Evaluation

Evaluates discovered sessions using Amazon Bedrock and writes results to CloudWatch.
"""

import boto3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import offline_eval_config as config
import time

def load_discovered_sessions() -> Dict[str, Any]:
    """Load discovered sessions from JSON file."""
    cfg = config.get_config()

    try:
        with open(cfg['discovered_sessions_file'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {cfg['discovered_sessions_file']}")
        print("Run 01_discover_sessions.py first!")
        return None
    except Exception as e:
        print(f"❌ Error loading sessions: {e}")
        return None

def create_evaluation_prompt(span: Dict[str, Any]) -> str:
    """Create evaluation prompt for a span."""

    attributes = span.get('attributes', {})
    events = span.get('events', [])

    # Extract user message and assistant response from events
    user_message = ""
    assistant_response = ""

    for event in events:
        event_attrs = event.get('attributes', {})
        if event_attrs.get('event.type') == 'user_message':
            user_message = event_attrs.get('content.preview', '')
        elif event_attrs.get('event.type') == 'assistant_response':
            assistant_response = event_attrs.get('content.preview', '')

    # Build context
    context = f"""
## Conversation Turn: {span['name']}

**User Request:**
{user_message}

**Assistant Response:**
{assistant_response}

**Context:**
- Model: {attributes.get('gen_ai.request.model', 'unknown')}
- Operation: {attributes.get('gen_ai.operation.name', 'unknown')}
- Tools Used: {attributes.get('tool.count', 0)}
- Turn Number: {attributes.get('ai.agent.turn_number', 'unknown')}

**Evaluation Task:**
{config.EVALUATION_RUBRIC}
"""

    return context

def evaluate_span_with_bedrock(span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Evaluate a single span using Amazon Bedrock."""
    cfg = config.get_config()
    bedrock = boto3.client('bedrock-runtime', region_name=cfg['aws_region'])

    try:
        # Create evaluation prompt
        prompt = create_evaluation_prompt(span)

        # Call Bedrock
        response = bedrock.invoke_model(
            modelId=cfg['bedrock_model_id'],
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0.0  # Deterministic evaluation
            })
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']

        # Extract JSON from response
        # Look for JSON block in the response
        try:
            # Try to find JSON in code blocks
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0].strip()
            else:
                # Try to parse the whole content as JSON
                json_str = content.strip()

            evaluation = json.loads(json_str)

            return {
                'score': float(evaluation.get('score', 0.0)),
                'reasoning': evaluation.get('reasoning', ''),
                'raw_response': content,
                'success': True
            }

        except Exception as parse_error:
            print(f"Warning: Could not parse evaluation JSON: {parse_error}")
            print(f"Raw response: {content[:200]}")
            return {
                'score': 0.0,
                'reasoning': f"Parse error: {str(parse_error)}",
                'raw_response': content,
                'success': False
            }

    except Exception as e:
        print(f"❌ Error evaluating span: {e}")
        import traceback
        traceback.print_exc()
        return None

def write_evaluation_to_cloudwatch(
    span: Dict[str, Any],
    evaluation: Dict[str, Any],
    session_id: str
) -> bool:
    """Write evaluation result to CloudWatch Logs in EMF format."""
    cfg = config.get_config()
    logs_client = boto3.client('logs', region_name=cfg['aws_region'])

    try:
        # Ensure log group exists
        try:
            logs_client.create_log_group(logGroupName=cfg['eval_results_log_group'])
            print(f"Created log group: {cfg['eval_results_log_group']}")
        except logs_client.exceptions.ResourceAlreadyExistsException:
            pass

        # Ensure log stream exists
        stream_name = f"eval-{session_id}"
        try:
            logs_client.create_log_stream(
                logGroupName=cfg['eval_results_log_group'],
                logStreamName=stream_name
            )
        except logs_client.exceptions.ResourceAlreadyExistsException:
            pass

        # Create EMF log entry
        emf_log = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": "AgentCore/Evaluation",
                    "Dimensions": [["ServiceName", "EvaluatorName"]],
                    "Metrics": [{
                        "Name": "EvaluationScore",
                        "Unit": "None"
                    }]
                }]
            },
            "ServiceName": cfg['service_name'],
            "EvaluatorName": cfg['evaluator_name'],
            "EvaluationScore": evaluation['score'],
            "SessionId": session_id,
            "TraceId": span.get('traceId'),
            "SpanId": span.get('spanId'),
            "SpanName": span.get('name'),
            "Reasoning": evaluation['reasoning'],
            "EvaluationTimestamp": datetime.now().isoformat(),
            "gen_ai.evaluation.name": cfg['evaluator_name'],
            "gen_ai.evaluation.score": evaluation['score'],
            "gen_ai.evaluation.reasoning": evaluation['reasoning']
        }

        # Write to CloudWatch
        logs_client.put_log_events(
            logGroupName=cfg['eval_results_log_group'],
            logStreamName=stream_name,
            logEvents=[{
                'timestamp': int(time.time() * 1000),
                'message': json.dumps(emf_log)
            }]
        )

        return True

    except Exception as e:
        print(f"❌ Error writing to CloudWatch: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_sessions(discovery_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate all sessions and their spans."""
    cfg = config.get_config()

    print(f"Evaluating {len(discovery_result['sessions'])} session(s)")
    print("=" * 70)
    print()

    all_results = []

    for session in discovery_result['sessions']:
        session_id = session['session_id']
        print(f"📊 Evaluating session: {session_id}")
        print(f"   Spans: {session['span_count']}")
        print()

        session_results = {
            'session_id': session_id,
            'span_count': session['span_count'],
            'evaluations': [],
            'average_score': 0.0
        }

        total_score = 0.0
        evaluated_count = 0

        for i, span in enumerate(session['spans'], 1):
            print(f"   Evaluating span {i}/{session['span_count']}: {span['name']}")

            # Evaluate the span
            evaluation = evaluate_span_with_bedrock(span)

            if evaluation and evaluation['success']:
                print(f"      Score: {evaluation['score']:.2f}")
                print(f"      Reasoning: {evaluation['reasoning'][:80]}...")

                # Write to CloudWatch
                write_success = write_evaluation_to_cloudwatch(span, evaluation, session_id)
                if write_success:
                    print(f"      ✅ Written to CloudWatch")
                else:
                    print(f"      ⚠️  Failed to write to CloudWatch")

                session_results['evaluations'].append({
                    'span_name': span['name'],
                    'trace_id': span['traceId'],
                    'span_id': span['spanId'],
                    'score': evaluation['score'],
                    'reasoning': evaluation['reasoning']
                })

                total_score += evaluation['score']
                evaluated_count += 1

            else:
                print(f"      ❌ Evaluation failed")

            print()

        # Calculate average
        if evaluated_count > 0:
            session_results['average_score'] = total_score / evaluated_count

        all_results.append(session_results)

        print(f"   Session average score: {session_results['average_score']:.2f}")
        print()

    return all_results

def save_evaluation_results(results: List[Dict[str, Any]]) -> None:
    """Save evaluation results to JSON file."""
    cfg = config.get_config()

    output = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'evaluator_name': cfg['evaluator_name'],
        'bedrock_model': cfg['bedrock_model_id'],
        'session_count': len(results),
        'results': results
    }

    with open(cfg['evaluation_results_file'], 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved evaluation results to: {cfg['evaluation_results_file']}")

def main():
    print("=" * 70)
    print("Claude Code Offline Evaluation - Session Evaluation")
    print("=" * 70)
    print()

    # Load discovered sessions
    discovery_result = load_discovered_sessions()
    if not discovery_result:
        return

    print(f"Loaded {len(discovery_result['sessions'])} session(s)")
    print(f"Discovery timestamp: {discovery_result['discovery_timestamp']}")
    print()

    # Evaluate sessions
    results = evaluate_sessions(discovery_result)

    # Save results
    save_evaluation_results(results)

    # Summary
    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    total_spans = sum(r['span_count'] for r in results)
    total_evaluated = sum(len(r['evaluations']) for r in results)
    avg_score = sum(r['average_score'] for r in results) / len(results) if results else 0.0

    print(f"Sessions evaluated: {len(results)}")
    print(f"Total spans: {total_spans}")
    print(f"Successfully evaluated: {total_evaluated}")
    print(f"Overall average score: {avg_score:.2f}")
    print()

    for result in results:
        print(f"Session: {result['session_id']}")
        print(f"  Average score: {result['average_score']:.2f}")
        print(f"  Evaluations: {len(result['evaluations'])}/{result['span_count']}")
        print()

    cfg = config.get_config()
    print(f"📊 View results in CloudWatch Logs:")
    print(f"   {cfg['eval_results_log_group']}")
    print("=" * 70)

if __name__ == "__main__":
    main()

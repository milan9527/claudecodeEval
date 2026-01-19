#!/usr/bin/env python3
"""
Session Discovery for Claude Code Offline Evaluation

Discovers sessions and spans from CloudWatch Logs for offline evaluation.
"""

import boto3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import offline_eval_config as config

def discover_sessions_from_cloudwatch() -> List[Dict[str, Any]]:
    """
    Discover sessions from CloudWatch Logs.

    Returns:
        List of session objects with session_id, span_count, and spans
    """
    cfg = config.get_config()
    logs_client = boto3.client('logs', region_name=cfg['aws_region'])

    print(f"Discovering sessions from: {cfg['source_log_group']}")
    print(f"Looking back: {cfg['lookback_hours']} hours")
    print("=" * 70)

    try:
        # Get log events from the stream
        response = logs_client.get_log_events(
            logGroupName=cfg['source_log_group'],
            logStreamName=cfg['source_log_stream'],
            startFromHead=False,
            limit=100  # Get last 100 events
        )

        events = response.get('events', [])
        print(f"Found {len(events)} total log events")

        # Parse events and group by session
        sessions = {}
        spans = []

        for event in events:
            try:
                log_data = json.loads(event['message'])

                # Check if this is a span (has name and kind)
                if 'name' in log_data and 'kind' in log_data:
                    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)

                    # Extract session ID from attributes
                    attributes = log_data.get('attributes', {})
                    session_id = attributes.get('session.id', 'unknown')

                    span_data = {
                        'name': log_data['name'],
                        'kind': log_data['kind'],
                        'traceId': log_data.get('traceId'),
                        'spanId': log_data.get('spanId'),
                        'timestamp': timestamp.isoformat(),
                        'attributes': attributes,
                        'events': log_data.get('events', []),
                        'raw_log': log_data
                    }

                    spans.append(span_data)

                    # Group by session
                    if session_id not in sessions:
                        sessions[session_id] = {
                            'session_id': session_id,
                            'span_count': 0,
                            'spans': [],
                            'first_seen': timestamp,
                            'last_seen': timestamp
                        }

                    sessions[session_id]['spans'].append(span_data)
                    sessions[session_id]['span_count'] += 1

                    if timestamp < sessions[session_id]['first_seen']:
                        sessions[session_id]['first_seen'] = timestamp
                    if timestamp > sessions[session_id]['last_seen']:
                        sessions[session_id]['last_seen'] = timestamp

            except json.JSONDecodeError:
                # Skip non-JSON events
                continue
            except Exception as e:
                print(f"Warning: Error parsing event: {e}")
                continue

        print(f"\nFound {len(spans)} OTEL spans")
        print(f"Grouped into {len(sessions)} session(s)")
        print()

        # Convert to list and add metadata
        session_list = []
        for session in sessions.values():
            session['first_seen'] = session['first_seen'].isoformat()
            session['last_seen'] = session['last_seen'].isoformat()
            session_list.append(session)

            print(f"Session: {session['session_id']}")
            print(f"  Spans: {session['span_count']}")
            print(f"  Period: {session['first_seen']} to {session['last_seen']}")
            print()

        return session_list

    except Exception as e:
        print(f"Error discovering sessions: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_discovered_sessions(sessions: List[Dict[str, Any]]) -> None:
    """Save discovered sessions to JSON file."""
    cfg = config.get_config()

    discovery_result = {
        'discovery_timestamp': datetime.now().isoformat(),
        'discovery_method': 'cloudwatch_direct',
        'source_log_group': cfg['source_log_group'],
        'lookback_hours': cfg['lookback_hours'],
        'session_count': len(sessions),
        'sessions': sessions
    }

    with open(cfg['discovered_sessions_file'], 'w') as f:
        json.dump(discovery_result, f, indent=2)

    print(f"✅ Saved {len(sessions)} session(s) to: {cfg['discovered_sessions_file']}")

def main():
    print("=" * 70)
    print("Claude Code Offline Evaluation - Session Discovery")
    print("=" * 70)
    print()

    # Discover sessions
    sessions = discover_sessions_from_cloudwatch()

    if not sessions:
        print("❌ No sessions found!")
        return

    # Save results
    save_discovered_sessions(sessions)

    # Summary
    print()
    print("=" * 70)
    print("Discovery Summary")
    print("=" * 70)

    total_spans = sum(s['span_count'] for s in sessions)
    print(f"Sessions discovered: {len(sessions)}")
    print(f"Total spans: {total_spans}")
    print(f"Average spans per session: {total_spans / len(sessions):.1f}")
    print()

    print("Next step: Run 02_evaluate_sessions.py to evaluate these sessions")
    print("=" * 70)

if __name__ == "__main__":
    main()

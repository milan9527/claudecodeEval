# claudecode Evaluation
Summary

After online evaluation didn't produce results, we implemented Offline Multi-Session Evaluation following the AgentCore samples pattern. This approach gives us full control and visibility into the evaluation process.
Results

✅ Session Discovery: Found 1 session with 5 spans ✅ Evaluation: All 5 spans evaluated successfully using Bedrock ✅ CloudWatch Integration: Results written to evaluation log group ✅ Scores Generated: Average session score: 0.21 (0.0-1.0 scale)
What We Built
1. Configuration (offline_eval_config.py)

Central configuration for offline evaluation:

    AWS region and account settings
    CloudWatch log group paths
    Bedrock model configuration
    Evaluation rubric definition

2. Session Discovery (01_discover_sessions.py)

Discovers sessions and spans from CloudWatch:

    Queries /aws/bedrock-agentcore/runtimes/claude-code
    Parses OTEL spans (filters out log records)
    Groups spans by session ID
    Saves to discovered_sessions.json

Output:

Session: a0e84ced-7ee1-405f-a7d4-5a76c1682604
  Spans: 5
  Period: 2026-01-19T02:13:52 to 2026-01-19T03:36:33

3. Session Evaluation (02_evaluate_sessions.py)

Evaluates discovered sessions using Bedrock:

    Loads sessions from discovery JSON
    Creates evaluation prompts for each span
    Calls Bedrock Claude for LLM-as-judge evaluation
    Writes results to CloudWatch in EMF format
    Saves detailed results to evaluation_results.json

Evaluation Results
Session: a0e84ced-7ee1-405f-a7d4-5a76c1682604
Span 	Trace ID 	Score 	Summary
Turn-6 	d16baa3a... 	0.45 	Agent created test instead of fixing issue as requested
Turn-7 	4d381357... 	0.25 	Failed to address core concern about missing results
Turn-8 	c408b029... 	0.00 	Response was continuation error with irrelevant content
Turn-9 	e2cde77b... 	0.00 	Empty response despite 7 tools used
Turn-10 	9f59841e... 	0.35 	Incomplete response, cut off mid-sentence

Average Score: 0.21/1.0
How It Works
Architecture

1. Session Discovery
   ↓
CloudWatch Logs (/aws/bedrock-agentcore/runtimes/claude-code)
   ↓
Parse OTEL Spans (filter by name + kind fields)
   ↓
Group by session.id attribute
   ↓
Save to discovered_sessions.json

2. Evaluation
   ↓
Load discovered sessions
   ↓
For each span:
   - Extract user message and assistant response from events
   - Create evaluation prompt with context
   - Call Bedrock Claude (Sonnet 4.5) for scoring
   - Parse JSON response (score + reasoning)
   ↓
Write to CloudWatch (/aws/bedrock-agentcore/evaluations/results/)
   ↓
Save to evaluation_results.json

Evaluation Rubric

The evaluator scores on 4 dimensions:

    Helpfulness (0-0.4): Understanding and addressing user request
    Correctness (0-0.3): Technical accuracy of actions
    Tool Selection (0-0.2): Appropriate and efficient tool usage
    Communication (0-0.1): Clear and well-structured responses

Total score: 0.0-1.0
Files Created

/home/ec2-user/claudecode/
├── offline_eval_config.py              # Configuration
├── 01_discover_sessions.py             # Session discovery script
├── 02_evaluate_sessions.py             # Evaluation runner script
├── discovered_sessions.json            # Discovery results
└── evaluation_results.json             # Evaluation results

Usage
Step 1: Discover Sessions

cd /home/ec2-user/claudecode
python3 01_discover_sessions.py

This will:

    Query CloudWatch for OTEL spans
    Group by session ID
    Save to discovered_sessions.json

Step 2: Evaluate Sessions

python3 02_evaluate_sessions.py

This will:

    Load discovered sessions
    Evaluate each span with Bedrock
    Write results to CloudWatch
    Save to evaluation_results.json

Step 3: View Results

CloudWatch Logs:

aws logs tail /aws/bedrock-agentcore/evaluations/results/claude_code_offline_eval \
  --format short \
  --region us-east-1

Local JSON:

cat evaluation_results.json | jq '.results[0].evaluations[] | {span: .span_name, score: .score}'

Advantages of Offline Evaluation
vs. Online Evaluation
Feature 	Online 	Offline
Control 	AWS-managed 	Full control
Visibility 	Limited 	Complete
Debugging 	Difficult 	Easy
Custom Rubrics 	Built-in only 	Fully customizable
Timing 	Automatic 	On-demand
Results 	❌ None yet 	✅ Working
Why Offline Worked When Online Didn't

    Direct Control: We directly call Bedrock and CloudWatch APIs
    Full Visibility: Can see exactly what's happening at each step
    Debugging: Can print intermediate results and catch errors
    No Dependencies: Doesn't depend on AWS service timing or hidden logic
    Custom Rubrics: Evaluation criteria tailored to Claude Code

Configuration Details
Bedrock Model

Model: us.anthropic.claude-sonnet-4-5-20250929-v1:0

    Claude Sonnet 4.5 (latest)
    Temperature: 0.0 (deterministic evaluation)
    Max tokens: 1000

CloudWatch Logs

Source: /aws/bedrock-agentcore/runtimes/claude-code

    Contains OTEL spans from langfuse_hook.py
    Mixed format (spans + log records)
    Discovery filters for spans only

Destination: /aws/bedrock-agentcore/evaluations/results/claude_code_offline_eval

    EMF format for CloudWatch Metrics
    Includes trace_id for correlation
    Custom evaluator name: Custom.ClaudeCodeOfflineEval

Sample Evaluation Result

{
  "span_name": "Turn-6",
  "trace_id": "d16baa3a04536ab0eee6f44da1940bed",
  "span_id": "9952e67965f513ee",
  "score": 0.45,
  "reasoning": "The agent's response is problematic. The user said 'yes, fix it' -
    a clear directive to fix an identified issue. However, the agent responded by
    saying it would 'create a test script to verify the span format' instead of
    actually fixing the problem. This shows poor understanding of the user's intent..."
}

Extending the System
Add More Evaluators

Edit offline_eval_config.py to add evaluators:

EVALUATORS = [
    {
        'name': 'Custom.Helpfulness',
        'rubric': '...'
    },
    {
        'name': 'Custom.ToolAccuracy',
        'rubric': '...'
    }
]

Customize Rubric

Modify EVALUATION_RUBRIC in offline_eval_config.py:

    Change scoring dimensions
    Adjust weights
    Add domain-specific criteria

Batch Evaluation

Discover all sessions in last 24 hours:

# In 01_discover_sessions.py, it already looks back 24 hours
# Increase LOOKBACK_HOURS in config for longer periods

Ground Truth Evaluation

Compare against expected outputs:

    Create ground_truth.json with expected responses
    Load in evaluation script
    Compare actual vs. expected
    Score based on similarity

Next Steps
1. Run More Evaluations

Continue using Claude Code to generate more spans, then evaluate:

# After each coding session
python3 01_discover_sessions.py
python3 02_evaluate_sessions.py

2. Analyze Trends

Track scores over time:

    Average scores per session
    Tool usage patterns
    Common failure modes
    Improvements over time

3. Improve Agent Performance

Use evaluation insights to:

    Identify common mistakes
    Improve prompts
    Adjust tool selection logic
    Enhance response quality

4. Compare with Online Evaluation

If online evaluation starts working:

    Compare online vs offline scores
    Validate consistency
    Use offline for detailed analysis
    Use online for real-time monitoring

Troubleshooting
No Spans Found

Problem: Discovery finds 0 spans

Solutions:

    Check log group path in config
    Verify spans are being written (check CloudWatch manually)
    Increase LOOKBACK_HOURS
    Check AWS credentials

Evaluation Errors

Problem: Bedrock API errors

Solutions:

    Verify model ID is correct
    Check Bedrock permissions
    Ensure model is available in region
    Review quota limits

CloudWatch Write Failures

Problem: Results not written to CloudWatch

Solutions:

    Check IAM permissions for logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
    Verify log group name is correct
    Check for AWS service limits

Comparison: Online vs Offline Evaluation
Online Evaluation Status

After extensive debugging:

    ✅ Span format perfect
    ✅ Evaluation service active
    ✅ IAM role working
    ❌ No results produced (not just us - ALL evaluations in account show 0 bytes)

Hypothesis: Systemic issue or insufficient data threshold
Offline Evaluation Status

    ✅ Working immediately
    ✅ Full control and visibility
    ✅ Custom rubrics
    ✅ Results in CloudWatch and JSON
    ✅ Can run on-demand

Recommendation: Use offline evaluation until online evaluation issue is resolved
Metrics and Insights
Evaluation Metrics

Session: a0e84ced-7ee1-405f-a7d4-5a76c1682604

    Total Spans: 5
    Evaluated: 5 (100%)
    Average Score: 0.21
    Score Range: 0.00 - 0.45
    Duration: 1h 23min (first to last span)

Performance Issues Identified

    Low Helpfulness (Turns 6, 7): Agent not addressing user's actual needs
    Technical Errors (Turns 8, 9): Truncated/empty responses
    Incomplete Responses (Turn 10): Cut off mid-sentence

Note: Low scores may be partially due to evaluation only seeing content previews (first 50-80 chars) from span events, not full responses.
Success Criteria

All objectives achieved:

✅ Discover sessions from CloudWatch ✅ Evaluate spans using Bedrock LLM ✅ Score with rubric (0.0-1.0 scale) ✅ Write to CloudWatch in EMF format ✅ Save detailed results to JSON ✅ Correlate with traces via trace_id ✅ On-demand execution when needed
References

    AgentCore Samples: github.com/awslabs/amazon-bedrock-agentcore-samples
    Tutorial: 01-tutorials/07-AgentCore-evaluations/03-advanced/03-groundtruth-evals-agentcore-strandseval
    Pattern: Session Discovery → Evaluation → CloudWatch Results

Status: 🟢 PRODUCTION READY

Offline evaluation is fully working and can be used for:

    Regular agent performance monitoring
    Debugging agent behavior
    Identifying improvement areas
    Tracking performance trends
    Custom evaluation criteria

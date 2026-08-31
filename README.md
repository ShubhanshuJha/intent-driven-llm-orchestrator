# AWS AI Agent

An AI-powered AWS agent that uses LLMs and tool calling to securely interact with, query, and manage AWS services through natural language.

## Overview

`aws-ai-agent` is a learning and development project focused on building an agentic interface for AWS.

The goal is to allow users to interact with AWS services using natural-language commands while the AI agent identifies the appropriate tool, executes the required AWS operation, and returns a useful response.

Example:

```text
User: Show me all running EC2 instances in ap-south-1.

AWS AI Agent:
→ Identifies the EC2 tool
→ Calls the AWS API
→ Retrieves running instances
→ Returns the result to the user
```

## Planned Architecture

```text
User
  |
  v
LLM / AI Agent
  |
  v
Intent & Tool Selection
  |
  v
AWS Tools
  |
  +-- S3
  +-- EC2
  +-- Lambda
  +-- CloudWatch
  +-- Glue
  +-- Athena
  +-- EMR
  +-- RDS
  +-- IAM
  +-- Other AWS Services
  |
  v
AWS SDK / boto3
  |
  v
AWS
```

## Key Goals

- Interact with AWS services using natural language
- Implement LLM-based tool calling
- Build reusable tools for different AWS services
- Use `boto3` for AWS API interactions
- Return structured AWS responses through the LLM
- Follow IAM least-privilege principles
- Introduce confirmation for sensitive or destructive operations
- Maintain auditability of agent actions

## Tech Stack

The project may use:

- Python
- AWS
- boto3
- LangChain
- LangGraph
- LLM APIs
- Tool Calling / Function Calling

## Example Use Cases

```text
"List all S3 buckets."

"Show running EC2 instances in us-east-1."

"Get the latest CloudWatch logs for this Lambda."

"Run this Athena query."

"Show the status of my Glue jobs."

"Start the Glue job named customer-data-processing."
```

## Security

AWS permissions should always follow the principle of least privilege.

Sensitive or destructive operations should require additional validation or user confirmation before execution.

Examples include:

- Terminating EC2 instances
- Deleting S3 objects or buckets
- Modifying IAM policies
- Deleting AWS resources
- Stopping production workloads

## Project Status

🚧 **Under Development**

This project is currently being built and additional AWS tools and agent capabilities will be added progressively.

## Roadmap

- [ ] Set up the base AI agent
- [ ] Configure AWS authentication
- [ ] Implement tool-calling architecture
- [ ] Add S3 tools
- [ ] Add EC2 tools
- [ ] Add CloudWatch tools
- [ ] Add Lambda tools
- [ ] Add Glue tools
- [ ] Add Athena tools
- [ ] Add EMR tools
- [ ] Add confirmation for destructive operations
- [ ] Add logging and auditing
- [ ] Add tests
- [ ] Add CLI / UI interface

## Getting Started

Clone the repository:

```bash
git clone https://github.com/<your-username>/aws-ai-agent.git
cd aws-ai-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

AWS and LLM configuration instructions will be added as the project evolves.

## Contributing

The project is currently in its early development stage. Contribution guidelines will be added later.

## License

License information will be added later.

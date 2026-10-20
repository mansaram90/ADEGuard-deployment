# Temporary ADEGuard AWS learning environment

This directory contains infrastructure as code, not evidence of a cloud deployment.
`aws-demo.yaml` prepares a disposable Docker host; application installation,
image selection, readiness and rollback remain separate steps. Preparing/linting
the files locally creates no AWS resources. Execute the AWS commands below only
after reviewing the account, region, cost, permissions, change set and cleanup.

The account was reported as a new Free plan with $100 credit and 183 days left.
Those are user-reported details, not an independently inspected balance or a
guarantee of eligibility. Do not upgrade the account to follow this exercise.
If AWS denies a service/instance or requests an upgrade, stop and review the plan.

## Resource design and code

```text
Your browser -> local Session Manager tunnel -> EC2 loopback:8080
  -> Nginx /api/predict -> FastAPI on the Compose network

EC2 -> outbound HTTPS -> AWS Systems Manager, package repositories and GHCR
```

The template defines 11 resources, all in a dedicated stack:

| Resource(s) | Purpose and key settings |
|---|---|
| VPC and subnet | A separate 10.42.0.0/24 network and 10.42.0.0/26 subnet, with DNS enabled. No existing/default VPC is modified. |
| Internet gateway and attachment | Outbound internet connectivity from this VPC. |
| Route table, internet route, subnet association | Route outbound traffic through the gateway; explicit dependencies establish this path before server startup. |
| Security group | No inbound rules. Outbound TCP 443 for package downloads, registry and SSM; VPC DNS resolution is available. |
| IAM role and instance profile | EC2 may assume the role with AWS's managed `AmazonSSMManagedInstanceCore` policy. No static AWS keys in the template, images or server startup script. |
| EC2 host | One Linux x86_64 `t3.small`, encrypted 30 GiB gp3 root volume, root volume/network interface deleted on termination, IMDSv2 required. |

`t3.small` (2 vCPU, 2 GiB RAM) matches AWS's documented new-account eligible
list and the existing x86_64 API image. The initially discussed `t3.medium`
is not used because it is absent from that eligibility list. Region/account
availability and quotas still require verification before launch. Do not choose
ARM/Graviton without publishing and verifying ARM-compatible application images.

The local API container used about 301.4 MiB at the sizing observation. That
supports trying this size for the tiny synthetic models; it is not a memory
peak, concurrency benchmark or guarantee against out-of-memory on AWS. Start
only API/frontend for the first exercise, not the full monitoring profile.

`CreditSpecification.CPUCredits: standard` avoids T3 surplus-credit billing;
CPU throughput can be throttled when credits are exhausted. It is not a general
spending cap. The instance requires IMDSv2 tokens with hop limit 1: SSM runs on
the host and the application needs no instance-role credentials. The managed
SSM policy is AWS's baseline agent policy, not an application-admin role; it is
not claimed to be the smallest possible custom policy. The operator's permissions
are separate from the EC2 instance role.

The subnet does not automatically assign public addresses to arbitrary future
instances. This host explicitly requests one auto-assigned public IPv4 to reach
SSM/package/registry endpoints through the internet gateway. This is a public
subnet with inbound traffic blocked, not a private subnet. The address is not
an Elastic IP and is released when the instance terminates. No NAT gateway,
load balancer, domain, database, private endpoint or public web listener is created.

`AmazonLinuxImage` is an AWS public SSM parameter resolved to a regional Amazon
Linux 2023 x86_64 AMI. `!Ref` connects resources by ID; `!Sub` inserts the stack
name/AWS partition into strings; `Fn::Base64` encodes the EC2 startup script.
Explicit `DependsOn` entries ensure the outbound route is ready before the
instance starts. AMI/package versions can change between deployments: record
the resolved `ImageId` output and installed Docker version. This is reproducible
infrastructure configuration, not a byte-identical operating-system build.

## Startup is not application deployment

EC2 user data runs as root on the initial boot. It creates `/opt/adeguard`,
installs Docker with Amazon Linux `dnf`, enables Docker and the preinstalled
SSM Agent, records `docker version`, then writes a completion marker.
`set -euo pipefail` stops on command failures; the final marker is not written
if installation fails. No model, Git repository, registry secret, Compose
plugin or app is installed by this template. Docker Engine and Docker Compose
are separate components; the pinned Compose plugin will be installed/validated
in the later application-deployment step.

CloudFormation does not wait for this script through a CreationPolicy signal.
`CREATE_COMPLETE` therefore means infrastructure was created, not that Docker
or ADEGuard is ready. Before app deployment, connect via SSM and check:

```bash
sudo cloud-init status --wait
sudo test -f /opt/adeguard/bootstrap-complete
sudo docker version
sudo systemctl is-active amazon-ssm-agent
```

On failure inspect `/var/log/cloud-init-output.log` and SSM Agent state/logs;
do not open SSH inbound as an automatic workaround. If SSM is unavailable,
review role attachment, agent, routing, outbound HTTPS and instance status from
the AWS Console. A failed bootstrap can leave billable infrastructure running;
delete the stack if abandoning the exercise.

## Estimated session cost

Planning example checked 20 October 2026: Linux on-demand `t3.small` in
`us-east-1` (Northern Virginia), two hours, 30 GiB baseline gp3 and one public
IPv4. No unused existing resources or other account activity are included.

| Component | Published reference rate | Two-hour estimate |
|---|---|---:|
| Compute | $0.0209/hour | $0.0418 |
| Public IPv4 | $0.005/hour | $0.0100 |
| Baseline gp3 | $0.08/GB-month; 30-day proration assumption | $0.0067 |
| Base total | Before credits/tax/data transfer/other usage | about $0.06 USD |

This replaces the earlier approximate $0.10 example for `t3.medium`. These
are planning figures, not a spending cap or a claim that the account will be
billed this exact amount. Recheck regional prices, actual usage and credit terms
before launching. Free plan/credits are not the old 750-hour allowance. Charges
can continue for retained resources; stopping a VM is not complete cleanup.
AWS Budgets notifications are delayed and do not by themselves cap spending.

Sources: [T3 pricing and CPU credits](https://aws.amazon.com/ec2/instance-types/t3/),
[IPv4 pricing](https://aws.amazon.com/vpc/pricing/),
[EBS pricing](https://aws.amazon.com/ebs/pricing/),
[new-account EC2 eligibility](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/LaunchingAndUsingInstances.html),
[budget behavior](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html).

## Local validation and CI

Install tooling in an isolated environment, from the repository root:

```powershell
python -m venv outputs/infra-tools
outputs/infra-tools/Scripts/python.exe -m pip install -r requirements-infra.txt
outputs/infra-tools/Scripts/cfn-lint.exe -t infra/aws-demo.yaml -r us-east-1
git diff --check
```

`cfn-lint` checks CloudFormation syntax, resource schemas, properties, references
and regional resource rules. It is not a live account permission, Free plan
eligibility, quota, bootstrap or connectivity test. The independent CI job
**CloudFormation template checks** runs the same pinned validator without AWS
credentials. Image build/publishing depends on it passing. No AWS deployment
or account authentication is performed by CI.

## Later AWS review and creation: do not run until approved

Use a non-root operator identity with MFA and the necessary deployment
permissions. Enable root MFA and keep root credentials for account-only tasks.
Configure authenticated AWS CLI access using an agreed temporary-credential
method, and install the local Session Manager plugin for port forwarding.
These workstation prerequisites are separate from this template. Do not send
credentials or MFA codes through chat. The commands assume region `us-east-1`;
do not create a duplicate stack elsewhere by changing region mid-exercise.

First check identity locally, service eligibility, instance quotas and the
published release. `.env.release` may still refer to pre-proxy images. Select
the two digests from a successful **Verify published release** run that includes
the Nginx routing change; do not use a floating `latest` tag or the held candidate.
Verify that GHCR images are publicly pullable or agree on scoped registry access.

For a future approved review, from the repository root:

```powershell
aws sts get-caller-identity
aws cloudformation validate-template --template-body file://infra/aws-demo.yaml --region us-east-1
aws cloudformation create-change-set --stack-name adeguard-learning --change-set-name reviewed-demo --change-set-type CREATE --template-body file://infra/aws-demo.yaml --capabilities CAPABILITY_IAM --region us-east-1 --tags Key=Project,Value=ADEGuard Key=Purpose,Value=temporary-learning
aws cloudformation wait change-set-create-complete --stack-name adeguard-learning --change-set-name reviewed-demo --region us-east-1
aws cloudformation describe-change-set --stack-name adeguard-learning --change-set-name reviewed-demo --region us-east-1
```

`validate-template` checks with AWS but creates no server. Creating a change
set records a plan and a REVIEW_IN_PROGRESS stack; it does not provision the
11 application infrastructure resources. Inspect the listed additions and
resolved AMI parameter. `CAPABILITY_IAM` acknowledges that the template creates
an unnamed IAM role/profile; it does not grant the operator missing permissions.
Local `--template-body` avoids the extra S3 staging bucket created by console
template upload. If using console upload instead, record the staging bucket/key
separately: that object/bucket is outside this template and is not automatically
removed with the stack. Do not delete staging objects belonging to other work.

Only after explicit launch approval, execute the reviewed change set:

```powershell
aws cloudformation execute-change-set --stack-name adeguard-learning --change-set-name reviewed-demo --region us-east-1
aws cloudformation wait stack-create-complete --stack-name adeguard-learning --region us-east-1
aws cloudformation describe-stacks --stack-name adeguard-learning --region us-east-1
```

Execution starts resource provisioning/usage. Preserve outputs and resource
IDs locally. On failure inspect CloudFormation Events; do not repeatedly create
additional stacks or assume failed creation consumed no resources/credits.

The later app exercise will install Compose, pull verified images, start the
release on loopback, check readiness/model identity/prediction through Nginx,
and rehearse an explicitly chosen operational rollback. It will use parent/demo
artifacts; HOLD remains in force for candidate promotion. Access will be through
an authenticated Session Manager tunnel, not a public HTTPS application.

Example future tunnel (replace the ID; keep this terminal running):

```powershell
aws ssm start-session --target i-REPLACE --document-name AWS-StartPortForwardingSession --parameters portNumber=8080,localPortNumber=18080 --region us-east-1
```

Then open `http://127.0.0.1:18080`. Same-origin `/api/predict` stays within that
tunnel and Nginx forwards internally; no second browser API tunnel is needed.
This is encrypted/authenticated tunnel access, not evidence of public HTTPS
configuration. Session Manager port forwarding requires the plugin and correct
operator/instance permissions; SSM registration alone is not app readiness.

## Cleanup after the exercise

Before deletion, capture only evidence needed for learning: stack/region,
instance and root-volume IDs, resolved AMI, selected image digests, readiness,
model identity, smoke results and rollback observations. Record the instance's
root-volume ID in EC2 before termination so it can be checked afterward.
Application container cleanup does not remove AWS infrastructure.

When finished with this specific disposable stack, request deletion and wait:

```powershell
aws cloudformation delete-stack --stack-name adeguard-learning --region us-east-1
aws cloudformation wait stack-delete-complete --stack-name adeguard-learning --region us-east-1
```

Check the same account/region for:

- Stack deletion completed, with no DELETE_FAILED resources remaining.
- Recorded instance terminated and recorded root volume deleted.
- Auto-assigned public IPv4 released; no Elastic IP was created by this template.
- Stack-owned security group, subnet, routes, gateway, VPC, role and profile removed.
- Any separately created upload/staging objects or other manual resources reviewed.
- Billing/credit usage checked later after reporting catches up.

Default stack deletion and root-volume `DeleteOnTermination` are intentional
for this disposable environment. No Retain or Snapshot policy is configured.
External changes/dependencies, permissions or a deletion failure can still leave
resources behind; verify rather than assume that clicking Delete completed.
Retain the local template, validation logs and learning notes for interviews.

Further references: [template and validation guide](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/template-guide.html),
[dependency ordering](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-attribute-dependson.html),
[SSM setup](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started.html),
[stack lifecycle](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/stacks.html),
[SSM port forwarding](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html).

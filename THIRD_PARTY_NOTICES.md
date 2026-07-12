# Third-Party Notices

## GIFT-Eval

TSRouter-VLDB uses the public GIFT-Eval benchmark as its forecasting workload and adapts its evaluation interfaces to run registered time-series foundation models through a unified workflow.

- Upstream dataset and benchmark: [Salesforce/GiftEval](https://huggingface.co/datasets/Salesforce/GiftEval)
- Upstream license: Apache-2.0
- Use in this repository: benchmark access, dataset layout compatibility, and evaluation integration

GIFT-Eval data are downloaded from the upstream repository at runtime and are not redistributed by this GitHub repository or its artifact bundles. Use of the benchmark remains subject to the upstream license, dataset card, and citation requirements.

## Model Implementations and Checkpoints

TSRouter-VLDB integrates official TSFM implementations and retrieves model checkpoints from their original public sources. Each model remains subject to its upstream license, model card, access controls, and citation requirements. Model checkpoints are not redistributed by this repository.

## Artifact Terms

The companion artifact repository contains derived results and reproduction inputs. Its `LICENSE`, Dataset Card, manifest, and checksums describe the terms and integrity information for those files.

# Articles

These articles show how to organize `bami` workflows for simulator-based
cognitive and behavioral models. Start with the end-to-end example if you are
new to the package, then use the concept articles when you need to adapt the
workflow to your own model.

## Start here

[Build and check a simple SDM workflow](sdm-fixed-simple.md)
: Build a fixed-trial simple workflow, simulate validation data, train or load
the workflow, sample posterior parameters, and check recovery.

## Core concepts

[Write a simulator for your own model](simulators.md)
: Learn the simulator contract and how to connect a custom behavioral model to
a `bami` workflow.

[Choose priors and links for model parameters](priors.md)
: Define estimated parameters, fixed simulator constants, distribution strings,
and link functions on the right scale.

## Workflow design

[Choose the right workflow structure](advanced-workflows.md)
: Compare fixed, flexible, simple, and hierarchical workflows so the workflow
matches the study design.

[Fit a hierarchical ezDM workflow](../examples/hierarchical-ezdm.md)
: Build, train, sample, and summarize a hierarchical aggregate-data workflow
with group and subject-level recovery checks.

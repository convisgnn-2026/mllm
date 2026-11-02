"""hetsim -- discrete-event simulator for a heterogeneous GPU pool serving
LLM / VLM / DiT tenants, plus the solver and router that sit on top of it.

Layers (see docs/simulator-design.md):
    spec      hardware and tenant descriptions (data only)
    model/    service-time models per tenant, calibrated per card
    core      event loop, instances, groups, router
    score     SLO / goodput / cost scoring of a finished run
    workload  arrival + request-class generators (TetriServe conventions)
"""

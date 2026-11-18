RKE2 (M9 data-center simulation) has no cloud secret manager and no Internet: the chart runs with
`secrets.mode: existing` and you create `ngc-api-secret` by hand during the connected phase (datacenter/bringup.sh).

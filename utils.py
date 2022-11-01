def rate(step, d_model, factor, warmup):
    if step == 0:
        step = 1
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step ** warmup ** (-1.5)))

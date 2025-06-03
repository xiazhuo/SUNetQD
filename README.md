# With Measurement Error

Toric 5x5:
Depolarizing    Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True))          [0.12, 0.03]    

Bias(5, 'Y')    Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True, ))        [0.12, 0.03]    

Bias(50, 'Y')   Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True, ))        [0.18, 0.06]    

Bit-phase-flip  Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True, ))        [0.15, 0.12]    


Toric 7x7:
Depolarizing    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.15, 0.12]    

Bias(5, 'Y')    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.15, 0.06]    

Bias(50, 'Y')   Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.15, 0.06]    

Bit-phase-flip  Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.15, 0.12]    


Toric 9x9:
Depolarizing    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.17, 0.12, 0.15]  

Bias(5, 'Y')    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.18, 0.06, 0.12, 0.04]

Bias(50, 'Y')   Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))    [0.15, 0.06, 0.04]

Bit-phase-flip  Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                   use_attn=(True, True, True))     [0.15, 0.12]    
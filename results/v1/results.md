## Article x FC (bottom level), test period

| model | subset | WAPE | MASE | Pinball | wQL | Cov80 |
|---|---|---|---|---|---|---|
| TFT | all | 0.4161 | 0.8162 | 2.2470 | 0.2640 | 0.7868 |
| TFT | warm | 0.4147 | 0.8162 | 2.2844 | 0.2633 | 0.7873 |
| TFT | cold-start | 0.4314 | nan | 1.9100 | 0.2726 | 0.7821 |
| MTFT-Picnic (10-d unimodal static) | all | 0.4119 | 0.8038 | 2.2312 | 0.2622 | 0.7938 |
| MTFT-Picnic (10-d unimodal static) | warm | 0.4108 | 0.8038 | 2.2696 | 0.2616 | 0.7951 |
| MTFT-Picnic (10-d unimodal static) | cold-start | 0.4251 | nan | 1.8852 | 0.2690 | 0.7824 |
| MTFT-SigLIP (static pooled) | all | 0.4214 | 0.8078 | 2.2890 | 0.2690 | 0.7876 |
| MTFT-SigLIP (static pooled) | warm | 0.4188 | 0.8078 | 2.3167 | 0.2670 | 0.7887 |
| MTFT-SigLIP (static pooled) | cold-start | 0.4502 | nan | 2.0398 | 0.2911 | 0.7774 |
| MTFT-SigLIP-XAttn (ours) | all | 0.4164 | 0.8018 | 2.2614 | 0.2657 | 0.7915 |
| MTFT-SigLIP-XAttn (ours) | warm | 0.4136 | 0.8018 | 2.2884 | 0.2637 | 0.7930 |
| MTFT-SigLIP-XAttn (ours) | cold-start | 0.4469 | nan | 2.0179 | 0.2880 | 0.7780 |

## Hierarchy (all levels), test period

| model | method | level | WAPE | MASE | wQL | Cov80 |
|---|---|---|---|---|---|---|
| TFT | base (incoherent) | national | 0.0465 | 1.6575 | 0.0316 | 0.4643 |
| TFT | base (incoherent) | fulfilment_centre | 0.0557 | 1.4049 | 0.0367 | 0.5536 |
| TFT | base (incoherent) | article_national | 0.2699 | 1.0385 | 0.1902 | 0.7057 |
| TFT | base (incoherent) | article_x_fc | 0.4161 | 0.8162 | 0.2640 | 0.7868 |
| TFT | bottom-up | national | 0.0779 | 2.7745 | 0.0558 | 0.0536 |
| TFT | bottom-up | fulfilment_centre | 0.0789 | 1.9048 | 0.0532 | 0.3571 |
| TFT | bottom-up | article_national | 0.2242 | 0.7484 | 0.1478 | 0.7482 |
| TFT | bottom-up | article_x_fc | 0.4161 | 0.8162 | 0.2640 | 0.7826 |
| TFT | MinT-shrink | national | 0.0379 | 1.3509 | 0.0235 | 0.5893 |
| TFT | MinT-shrink | fulfilment_centre | 0.0435 | 1.0832 | 0.0272 | 0.7143 |
| TFT | MinT-shrink | article_national | 0.2362 | 0.7324 | 0.1628 | 0.7357 |
| TFT | MinT-shrink | article_x_fc | 0.4381 | 0.8439 | 0.2797 | 0.7522 |
| MTFT-Picnic (10-d unimodal static) | base (incoherent) | national | 0.0465 | 1.6575 | 0.0316 | 0.4643 |
| MTFT-Picnic (10-d unimodal static) | base (incoherent) | fulfilment_centre | 0.0557 | 1.4049 | 0.0367 | 0.5536 |
| MTFT-Picnic (10-d unimodal static) | base (incoherent) | article_national | 0.2694 | 1.0385 | 0.1899 | 0.7059 |
| MTFT-Picnic (10-d unimodal static) | base (incoherent) | article_x_fc | 0.4119 | 0.8038 | 0.2622 | 0.7938 |
| MTFT-Picnic (10-d unimodal static) | bottom-up | national | 0.0912 | 3.2489 | 0.0685 | 0.0357 |
| MTFT-Picnic (10-d unimodal static) | bottom-up | fulfilment_centre | 0.0922 | 2.2198 | 0.0645 | 0.2643 |
| MTFT-Picnic (10-d unimodal static) | bottom-up | article_national | 0.2246 | 0.7275 | 0.1480 | 0.7534 |
| MTFT-Picnic (10-d unimodal static) | bottom-up | article_x_fc | 0.4119 | 0.8038 | 0.2622 | 0.7890 |
| MTFT-Picnic (10-d unimodal static) | MinT-shrink | national | 0.0417 | 1.4862 | 0.0259 | 0.5714 |
| MTFT-Picnic (10-d unimodal static) | MinT-shrink | fulfilment_centre | 0.0464 | 1.1558 | 0.0290 | 0.6786 |
| MTFT-Picnic (10-d unimodal static) | MinT-shrink | article_national | 0.2328 | 0.7185 | 0.1606 | 0.7457 |
| MTFT-Picnic (10-d unimodal static) | MinT-shrink | article_x_fc | 0.4311 | 0.8280 | 0.2758 | 0.7630 |
| MTFT-SigLIP (static pooled) | base (incoherent) | national | 0.0465 | 1.6575 | 0.0316 | 0.4643 |
| MTFT-SigLIP (static pooled) | base (incoherent) | fulfilment_centre | 0.0557 | 1.4049 | 0.0367 | 0.5536 |
| MTFT-SigLIP (static pooled) | base (incoherent) | article_national | 0.2716 | 1.0385 | 0.1912 | 0.7048 |
| MTFT-SigLIP (static pooled) | base (incoherent) | article_x_fc | 0.4214 | 0.8078 | 0.2690 | 0.7876 |
| MTFT-SigLIP (static pooled) | bottom-up | national | 0.1175 | 4.1866 | 0.0948 | 0.0000 |
| MTFT-SigLIP (static pooled) | bottom-up | fulfilment_centre | 0.1187 | 2.8658 | 0.0888 | 0.0929 |
| MTFT-SigLIP (static pooled) | bottom-up | article_national | 0.2410 | 0.7375 | 0.1594 | 0.7437 |
| MTFT-SigLIP (static pooled) | bottom-up | article_x_fc | 0.4214 | 0.8078 | 0.2690 | 0.7836 |
| MTFT-SigLIP (static pooled) | MinT-shrink | national | 0.0447 | 1.5939 | 0.0275 | 0.4464 |
| MTFT-SigLIP (static pooled) | MinT-shrink | fulfilment_centre | 0.0494 | 1.2431 | 0.0308 | 0.6250 |
| MTFT-SigLIP (static pooled) | MinT-shrink | article_national | 0.2349 | 0.7198 | 0.1620 | 0.7430 |
| MTFT-SigLIP (static pooled) | MinT-shrink | article_x_fc | 0.4330 | 0.8312 | 0.2791 | 0.7593 |
| MTFT-SigLIP-XAttn (ours) | base (incoherent) | national | 0.0465 | 1.6575 | 0.0316 | 0.4643 |
| MTFT-SigLIP-XAttn (ours) | base (incoherent) | fulfilment_centre | 0.0557 | 1.4049 | 0.0367 | 0.5536 |
| MTFT-SigLIP-XAttn (ours) | base (incoherent) | article_national | 0.2717 | 1.0385 | 0.1911 | 0.7047 |
| MTFT-SigLIP-XAttn (ours) | base (incoherent) | article_x_fc | 0.4164 | 0.8018 | 0.2657 | 0.7915 |
| MTFT-SigLIP-XAttn (ours) | bottom-up | national | 0.0978 | 3.4845 | 0.0751 | 0.0000 |
| MTFT-SigLIP-XAttn (ours) | bottom-up | fulfilment_centre | 0.0983 | 2.3615 | 0.0696 | 0.2321 |
| MTFT-SigLIP-XAttn (ours) | bottom-up | article_national | 0.2327 | 0.7205 | 0.1541 | 0.7504 |
| MTFT-SigLIP-XAttn (ours) | bottom-up | article_x_fc | 0.4164 | 0.8018 | 0.2657 | 0.7869 |
| MTFT-SigLIP-XAttn (ours) | MinT-shrink | national | 0.0462 | 1.6474 | 0.0285 | 0.4821 |
| MTFT-SigLIP-XAttn (ours) | MinT-shrink | fulfilment_centre | 0.0494 | 1.2338 | 0.0310 | 0.6429 |
| MTFT-SigLIP-XAttn (ours) | MinT-shrink | article_national | 0.2345 | 0.7074 | 0.1618 | 0.7472 |
| MTFT-SigLIP-XAttn (ours) | MinT-shrink | article_x_fc | 0.4317 | 0.8250 | 0.2776 | 0.7631 |

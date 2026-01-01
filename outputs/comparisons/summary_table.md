| Group        | Protocol        | Model                        |   Accuracy |     F1 |    AUC |
|:-------------|:----------------|:-----------------------------|-----------:|-------:|-------:|
| simclr       | simclr_finetune | densenet121                  |     0.9878 | 0.9867 | 0.9993 |
| transformers | supervised      | swin_tiny_patch4_window7_224 |     0.9854 | 0.984  | 0.9985 |
| cnns         | supervised      | densenet121                  |     0.983  | 0.9814 | 0.9992 |
| cnns         | supervised      | efficientnet_b0              |     0.983  | 0.9814 | 0.9989 |
| simclr       | simclr_linear   | densenet121                  |     0.9805 | 0.9785 | 0.9991 |
| simclr       | simclr_finetune | swin_tiny_patch4_window7_224 |     0.9805 | 0.9784 | 0.9991 |
| cnns         | supervised      | resnet50                     |     0.9781 | 0.976  | 0.9972 |
| transformers | supervised      | vit_base_patch16_224         |     0.9708 | 0.9681 | 0.9967 |
| simclr       | simclr_linear   | swin_tiny_patch4_window7_224 |     0.9659 | 0.9628 | 0.9974 |
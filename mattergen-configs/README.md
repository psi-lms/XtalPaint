# Configuration files to sample and train MatterGen

```
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
Adapted from https://github.com/microsoft/mattergen/tree/main/mattergen/conf
    and https://github.com/microsoft/mattergen/tree/main/sampling_conf
```

The `conf` subdirectory contains the standard configurations for training (see [here](https://github.com/microsoft/mattergen/tree/main/mattergen/conf)) and the additional ones for the `pos-only` and the `TD-paint` compatible model. We also copied the standard ones from the `MatterGen` repository to make it easier to specify the configuration file, as the new configuration files only overwrite a few parameters compared to the standard ones.

Similarly, `sampling_conf` contains the standard sampling configurations (see [here](https://github.com/microsoft/mattergen/tree/main/sampling_conf)) and the additional one for the `TD-paint` compatible model.



To perform the training, one can for example run the following command:

```bash
mattergen-train --config-name=inpaint ~trainer.logger data_module=alex_mp_20
```
See the `MatterGen` repository for more details on the training procedure.

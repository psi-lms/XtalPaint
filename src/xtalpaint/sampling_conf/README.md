# Configuration files to sample and train MatterGen

```
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
Adapted from https://github.com/microsoft/mattergen/tree/main/mattergen/conf
    and https://github.com/microsoft/mattergen/tree/main/sampling_conf
```


The sampling configurations (the standard ones copied from the `MatterGen` repository, see [here](https://github.com/microsoft/mattergen/tree/main/sampling_conf), plus the additional one for the `TD-paint` compatible model) are shipped inside the package at `src/xtalpaint/sampling_conf`, so that they are also available for wheel installs. XtalPaint falls back to them automatically when `sampling_config_path` is not set, because `MatterGen`'s own default resolves relative to its install location and only exists for source checkouts.

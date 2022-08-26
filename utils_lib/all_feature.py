import sys
sys.path.append('../utils_lib')

import utils_lib.feature_extractor_layers as FeatureExtractorLayer #a little useless but better for refactor

all_feature_extract = [
   
    {#0
        "feature_layer"        : FeatureExtractorLayer.NoneLayer,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [0],
        "name"                 : "none",
        "description"          : "no operation\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    
    ################         COS   #####################
    {#1
        "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer_cos,
        "output_feature_nb"    : lambda order,input: ((order+1)**input),
        "order"                : [4],
        "name"                 : "dff",
        "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#2
        "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [8,16,64], #4/8>16>>all 8
        "name"                 : "dflf_ll",
        "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,        
        "color"                : None,    
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#3
        "feature_layer"        : FeatureExtractorLayer.D_FLF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [8,16,64],
        "name"                 : "dflf",
        "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#4
        "feature_layer"        : FeatureExtractorLayer.R_FF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,256],#work on 4 #warnuing warning, order r_ff != order d_ff
        "name"                 : "rff",
        "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#5
        "feature_layer"        : FeatureExtractorLayer.R_FLF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [8,16], #work on 4
        "name"                 : "rflf",
        "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#6
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#7
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#8
        "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [8,16], #16,32,64,128
        "name"                 : "rflfnni",
        "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#9
        "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [8,16],
        "name"                 : "dflfnni",
        "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#10
        "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [8,16],#work on 4
        "name"                 : "lflfnni",
        "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#11
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_cheat,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_bet",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#12
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_cheat,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_bet",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },

]
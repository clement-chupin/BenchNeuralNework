
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


sys.path.append(os.path.join(os.path.dirname(__file__),'../utils_lib'))

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
    {#1
        "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer,
        "output_feature_nb"    : lambda order,input: ((order+1)**input),
        "order"                : [4],
        "name"                 : "dff",
        "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#2
        "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #4/8>16>>all 8
        "name"                 : "dflf_ll",
        "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,        
        "color"                : None,    
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#3
        "feature_layer"        : FeatureExtractorLayer.D_FLF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflf",
        "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#4
        "feature_layer"        : FeatureExtractorLayer.R_FF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#work on 4 #warnuing warning, order r_ff != order d_ff
        "name"                 : "rff",
        "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#5
        "feature_layer"        : FeatureExtractorLayer.R_FLF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #work on 4
        "name"                 : "rflf",
        "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#6
        "feature_layer"        : FeatureExtractorLayer.L_FF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#7
        "feature_layer"        : FeatureExtractorLayer.L_FLF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#,128*0],
        "name"                 : "lflf",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#8
        "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #16,32,64,128
        "name"                 : "rflfnni",
        "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#9
        "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflfnni",
        "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#10
        "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#work on 4
        "name"                 : "lflfnni",
        "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },


    ################         COS   #####################
    {#1
        "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer_cos,
        "output_feature_nb"    : lambda order,input: ((order+1)**input),
        "order"                : [4],
        "name"                 : "dff+cos",
        "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#2
        "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #4/8>16>>all 8
        "name"                 : "dflf_ll+cos",
        "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,        
        "color"                : None,    
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#3
        "feature_layer"        : FeatureExtractorLayer.D_FLF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflf+cos",
        "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#4
        "feature_layer"        : FeatureExtractorLayer.R_FF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#work on 4 #warnuing warning, order r_ff != order d_ff
        "name"                 : "rff+cos",
        "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#5
        "feature_layer"        : FeatureExtractorLayer.R_FLF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #work on 4
        "name"                 : "rflf+cos",
        "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#6
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff+cos",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#7
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#,128*0],
        "name"                 : "lflf+cos",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#8
        "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #16,32,64,128
        "name"                 : "rflfnni+cos",
        "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#9
        "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflfnni+cos",
        "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#10
        "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI_cos,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#work on 4
        "name"                 : "lflfnni+cos",
        "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },

    #######################    SIN   ###########################

    {#1
        "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer_sin,
        "output_feature_nb"    : lambda order,input: ((order+1)**input),
        "order"                : [4],
        "name"                 : "dff+sin",
        "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#2
        "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #4/8>16>>all 8
        "name"                 : "dflf_ll+sin",
        "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,        
        "color"                : None,    
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#3
        "feature_layer"        : FeatureExtractorLayer.D_FLF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflf+sin",
        "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#4
        "feature_layer"        : FeatureExtractorLayer.R_FF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#work on 4 #warnuing warning, order r_ff != order d_ff
        "name"                 : "rff+sin",
        "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#5
        "feature_layer"        : FeatureExtractorLayer.R_FLF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #work on 4
        "name"                 : "rflf+sin",
        "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#6
        "feature_layer"        : FeatureExtractorLayer.L_FF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff+sin",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#7
        "feature_layer"        : FeatureExtractorLayer.L_FLF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#,128*0],
        "name"                 : "lflf+sin",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#8
        "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #16,32,64,128
        "name"                 : "rflfnni+sin",
        "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#9
        "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflfnni+sin",
        "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#10
        "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#work on 4
        "name"                 : "lflfnni+sin",
        "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },

    ######################### SIN +COS +[-1,1]
    {#0
        "feature_layer"        : FeatureExtractorLayer.NoneLayer,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [0],
        "name"                 : "none_oss",
        "description"          : "no operation\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#1
        "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer,
        "output_feature_nb"    : lambda order,input: ((order+1)**input),
        "order"                : [4],
        "name"                 : "dff_oss",
        "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#2
        "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #4/8>16>>all 8
        "name"                 : "dflf_ll_oss",
        "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,        
        "color"                : None,    
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#3
        "feature_layer"        : FeatureExtractorLayer.D_FLF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflf_oss",
        "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#4
        "feature_layer"        : FeatureExtractorLayer.R_FF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#work on 4 #warnuing warning, order r_ff != order d_ff
        "name"                 : "rff_oss",
        "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#5
        "feature_layer"        : FeatureExtractorLayer.R_FLF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #work on 4
        "name"                 : "rflf_oss",
        "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#6
        "feature_layer"        : FeatureExtractorLayer.L_FF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_oss",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#7
        "feature_layer"        : FeatureExtractorLayer.L_FLF,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#,128*0],
        "name"                 : "lflf_oss",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#8
        "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #16,32,64,128
        "name"                 : "rflfnni_oss",
        "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#9
        "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflfnni_oss",
        "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#10
        "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#work on 4
        "name"                 : "lflfnni_oss",
        "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    ######################### SIN +[-1,1]
    {#1
        "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer_sin,
        "output_feature_nb"    : lambda order,input: ((order+1)**input),
        "order"                : [4],
        "name"                 : "dff+sin_oss",
        "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#2
        "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #4/8>16>>all 8
        "name"                 : "dflf_ll+sin_oss",
        "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,        
        "color"                : None,    
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#3
        "feature_layer"        : FeatureExtractorLayer.D_FLF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflf+sin_oss",
        "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#4
        "feature_layer"        : FeatureExtractorLayer.R_FF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#work on 4 #warnuing warning, order r_ff != order d_ff
        "name"                 : "rff+sin_oss",
        "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#5
        "feature_layer"        : FeatureExtractorLayer.R_FLF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #work on 4
        "name"                 : "rflf+sin_oss",
        "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#6
        "feature_layer"        : FeatureExtractorLayer.L_FF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [32,64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff+sin_oss",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#7
        "feature_layer"        : FeatureExtractorLayer.L_FLF_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#,128*0],
        "name"                 : "lflf+sin_oss",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#8
        "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32], #16,32,64,128
        "name"                 : "rflfnni+sin_oss",
        "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#9
        "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],
        "name"                 : "dflfnni+sin_oss",
        "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
    {#10
        "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI_sin,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,32],#work on 4
        "name"                 : "lflfnni+sin_oss",
        "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
        "power"                : 0,
        "color"                : "#009",
        "obs_shape"            : {"range":1.0,"offset":0.0},
    },
]
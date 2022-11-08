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
    {#13
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_weird,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_wei",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#14
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_weird,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_wei",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#15
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_sig,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_a",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#16
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_sig,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_a",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#17
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_relu,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_b",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#18
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_b",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#19
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_nude,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_c",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#20
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_nude,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_c",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#21
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_genius,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#,128*0],
        "name"                 : "lff_gen",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#22
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_genius,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16],#,128*0],
        "name"                 : "lflf_gen",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#23
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_stupid,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#,128*0],
        "name"                 : "lff_stup",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#24
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_stupid,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16],#,128*0],
        "name"                 : "lflf_stup",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },




    {#25
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_genius_a,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#,128*0],
        "name"                 : "lff_gen_a",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#26
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_genius_a,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16],#,128*0],
        "name"                 : "lflf_gen_a",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#27
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_stupid_a,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#,128*0],
        "name"                 : "lff_stup_a",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#28
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_stupid_a,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16],#,128*0],
        "name"                 : "lflf_stup_a",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },




    {#29
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_relu_a,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_ba",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#30
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_a,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_ba",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#31
        "feature_layer"        : FeatureExtractorLayer.L_FF_cos_relu_b,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_bb",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#32
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_b,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_bb",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#33
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_c,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_bc",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#34
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_c,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_bc",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#35
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_d,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_bd",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#36
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_d,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_bd",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#37
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_e,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [64,128,256],#256,512],#warnuing warning, order r_ff != order d_ff
        "name"                 : "lff_be",
        "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,

        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#38
        "feature_layer"        : FeatureExtractorLayer.L_FLF_cos_relu_e,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [4,8,16,64],#,128*0],
        "name"                 : "lflf_be",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#39
        "feature_layer"        : FeatureExtractorLayer.outsider,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [2,4,8,16,32],#,128*0],
        "name"                 : "outsider",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#40
        "feature_layer"        : FeatureExtractorLayer.outsider2,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [2,4,8,16,32],#,128*0],
        "name"                 : "outsiderv",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#41
        "feature_layer"        : FeatureExtractorLayer.outsider3,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [2,4,8,16,32],#,128*0],
        "name"                 : "outsiderb",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },
    {#42
        "feature_layer"        : FeatureExtractorLayer.FFP,
        "output_feature_nb"    : lambda order,input: (order + input),
        "order"                : [2,4,8],#,128*0],
        "name"                 : "FFP",
        "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
        "power"                : 0,
        "color"                : None,
        "obs_shape"            : {"range":0.5,"offset":0.5},
    },


    

]
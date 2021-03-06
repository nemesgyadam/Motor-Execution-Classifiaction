??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

;
Elu
features"T
activations"T"
Ttype:
2
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_48/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_48/gamma
?
0batch_normalization_48/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_48/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_48/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_48/beta
?
/batch_normalization_48/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_48/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_48/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_48/moving_mean
?
6batch_normalization_48/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_48/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_48/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_48/moving_variance
?
:batch_normalization_48/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_48/moving_variance*
_output_shapes
:*
dtype0
?
$depthwise_conv2d_10/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$depthwise_conv2d_10/depthwise_kernel
?
8depthwise_conv2d_10/depthwise_kernel/Read/ReadVariableOpReadVariableOp$depthwise_conv2d_10/depthwise_kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_49/gamma
?
0batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_49/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_49/beta
?
/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_49/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_49/moving_mean
?
6batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_49/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_49/moving_variance
?
:batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_49/moving_variance*
_output_shapes
:*
dtype0
?
$separable_conv2d_10/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_10/depthwise_kernel
?
8separable_conv2d_10/depthwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_10/depthwise_kernel*&
_output_shapes
:*
dtype0
?
$separable_conv2d_10/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$separable_conv2d_10/pointwise_kernel
?
8separable_conv2d_10/pointwise_kernel/Read/ReadVariableOpReadVariableOp$separable_conv2d_10/pointwise_kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_50/gamma
?
0batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_50/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_50/beta
?
/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_50/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_50/moving_mean
?
6batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_50/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_50/moving_variance
?
:batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_50/moving_variance*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:0*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
SGD/conv2d_28/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/conv2d_28/kernel/momentum
?
1SGD/conv2d_28/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_28/kernel/momentum*&
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_48/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_48/gamma/momentum
?
=SGD/batch_normalization_48/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_48/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_48/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_48/beta/momentum
?
<SGD/batch_normalization_48/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_48/beta/momentum*
_output_shapes
:*
dtype0
?
1SGD/depthwise_conv2d_10/depthwise_kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31SGD/depthwise_conv2d_10/depthwise_kernel/momentum
?
ESGD/depthwise_conv2d_10/depthwise_kernel/momentum/Read/ReadVariableOpReadVariableOp1SGD/depthwise_conv2d_10/depthwise_kernel/momentum*&
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_49/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_49/gamma/momentum
?
=SGD/batch_normalization_49/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_49/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_49/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_49/beta/momentum
?
<SGD/batch_normalization_49/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_49/beta/momentum*
_output_shapes
:*
dtype0
?
1SGD/separable_conv2d_10/depthwise_kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31SGD/separable_conv2d_10/depthwise_kernel/momentum
?
ESGD/separable_conv2d_10/depthwise_kernel/momentum/Read/ReadVariableOpReadVariableOp1SGD/separable_conv2d_10/depthwise_kernel/momentum*&
_output_shapes
:*
dtype0
?
1SGD/separable_conv2d_10/pointwise_kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31SGD/separable_conv2d_10/pointwise_kernel/momentum
?
ESGD/separable_conv2d_10/pointwise_kernel/momentum/Read/ReadVariableOpReadVariableOp1SGD/separable_conv2d_10/pointwise_kernel/momentum*&
_output_shapes
:*
dtype0
?
)SGD/batch_normalization_50/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_50/gamma/momentum
?
=SGD/batch_normalization_50/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_50/gamma/momentum*
_output_shapes
:*
dtype0
?
(SGD/batch_normalization_50/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_50/beta/momentum
?
<SGD/batch_normalization_50/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_50/beta/momentum*
_output_shapes
:*
dtype0
?
SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0**
shared_nameSGD/dense/kernel/momentum
?
-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes

:0*
dtype0
?
SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
?T
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?T
value?SB?S B?S
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
 moving_variance
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h
%depthwise_kernel
&	variables
'regularization_losses
(trainable_variables
)	keras_api
?
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
R
;	variables
<regularization_losses
=trainable_variables
>	keras_api
~
?depthwise_kernel
@pointwise_kernel
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
R
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
R
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
h

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
R
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
?
hiter
	idecay
jmomentummomentum?momentum?momentum?%momentum?+momentum?,momentum??momentum?@momentum?Fmomentum?Gmomentum?^momentum?_momentum?
?
0
1
2
3
 4
%5
+6
,7
-8
.9
?10
@11
F12
G13
H14
I15
^16
_17
 
V
0
1
2
%3
+4
,5
?6
@7
F8
G9
^10
_11
?
	variables
klayer_metrics
lmetrics
mnon_trainable_variables

nlayers
regularization_losses
trainable_variables
olayer_regularization_losses
 
\Z
VARIABLE_VALUEconv2d_28/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
	variables
player_metrics
qmetrics
rnon_trainable_variables

slayers
regularization_losses
trainable_variables
tlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_48/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_48/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_48/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_48/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 3
 

0
1
?
!	variables
ulayer_metrics
vmetrics
wnon_trainable_variables

xlayers
"regularization_losses
#trainable_variables
ylayer_regularization_losses
zx
VARIABLE_VALUE$depthwise_conv2d_10/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

%0
 

%0
?
&	variables
zlayer_metrics
{metrics
|non_trainable_variables

}layers
'regularization_losses
(trainable_variables
~layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_49/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_49/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_49/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_49/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
-2
.3
 

+0
,1
?
/	variables
layer_metrics
?metrics
?non_trainable_variables
?layers
0regularization_losses
1trainable_variables
 ?layer_regularization_losses
 
 
 
?
3	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
4regularization_losses
5trainable_variables
 ?layer_regularization_losses
 
 
 
?
7	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
8regularization_losses
9trainable_variables
 ?layer_regularization_losses
 
 
 
?
;	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
<regularization_losses
=trainable_variables
 ?layer_regularization_losses
zx
VARIABLE_VALUE$separable_conv2d_10/depthwise_kernel@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$separable_conv2d_10/pointwise_kernel@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
?
A	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Bregularization_losses
Ctrainable_variables
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_50/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_50/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_50/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_50/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
H2
I3
 

F0
G1
?
J	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Kregularization_losses
Ltrainable_variables
 ?layer_regularization_losses
 
 
 
?
N	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Oregularization_losses
Ptrainable_variables
 ?layer_regularization_losses
 
 
 
?
R	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Sregularization_losses
Ttrainable_variables
 ?layer_regularization_losses
 
 
 
?
V	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Wregularization_losses
Xtrainable_variables
 ?layer_regularization_losses
 
 
 
?
Z	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
[regularization_losses
\trainable_variables
 ?layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
?
`	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
aregularization_losses
btrainable_variables
 ?layer_regularization_losses
 
 
 
?
d	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
eregularization_losses
ftrainable_variables
 ?layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
*
0
 1
-2
.3
H4
I5
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
 
 
 
 
 
 

0
 1
 
 
 
 
 
 
 
 
 

-0
.1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

H0
I1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUESGD/conv2d_28/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_48/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_48/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1SGD/depthwise_conv2d_10/depthwise_kernel/momentumclayer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_49/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_49/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1SGD/separable_conv2d_10/depthwise_kernel/momentumclayer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1SGD/separable_conv2d_10/pointwise_kernel/momentumclayer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)SGD/batch_normalization_50/gamma/momentumXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/batch_normalization_50/beta/momentumWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_11Placeholder*/
_output_shapes
:?????????d*
dtype0*$
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11conv2d_28/kernelbatch_normalization_48/gammabatch_normalization_48/beta"batch_normalization_48/moving_mean&batch_normalization_48/moving_variance$depthwise_conv2d_10/depthwise_kernelbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_variance$separable_conv2d_10/depthwise_kernel$separable_conv2d_10/pointwise_kernelbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_variancedense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_11055819
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices$conv2d_28/kernel/Read/ReadVariableOp0batch_normalization_48/gamma/Read/ReadVariableOp/batch_normalization_48/beta/Read/ReadVariableOp6batch_normalization_48/moving_mean/Read/ReadVariableOp:batch_normalization_48/moving_variance/Read/ReadVariableOp8depthwise_conv2d_10/depthwise_kernel/Read/ReadVariableOp0batch_normalization_49/gamma/Read/ReadVariableOp/batch_normalization_49/beta/Read/ReadVariableOp6batch_normalization_49/moving_mean/Read/ReadVariableOp:batch_normalization_49/moving_variance/Read/ReadVariableOp8separable_conv2d_10/depthwise_kernel/Read/ReadVariableOp8separable_conv2d_10/pointwise_kernel/Read/ReadVariableOp0batch_normalization_50/gamma/Read/ReadVariableOp/batch_normalization_50/beta/Read/ReadVariableOp6batch_normalization_50/moving_mean/Read/ReadVariableOp:batch_normalization_50/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1SGD/conv2d_28/kernel/momentum/Read/ReadVariableOp=SGD/batch_normalization_48/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_48/beta/momentum/Read/ReadVariableOpESGD/depthwise_conv2d_10/depthwise_kernel/momentum/Read/ReadVariableOp=SGD/batch_normalization_49/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_49/beta/momentum/Read/ReadVariableOpESGD/separable_conv2d_10/depthwise_kernel/momentum/Read/ReadVariableOpESGD/separable_conv2d_10/pointwise_kernel/momentum/Read/ReadVariableOp=SGD/batch_normalization_50/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_50/beta/momentum/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOpConst"/device:CPU:0*4
dtypes*
(2&	
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-4/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-4/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOpAssignVariableOpconv2d_28/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_1AssignVariableOpbatch_normalization_48/gamma
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_2AssignVariableOpbatch_normalization_48/beta
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_3AssignVariableOp"batch_normalization_48/moving_mean
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
v
AssignVariableOp_4AssignVariableOp&batch_normalization_48/moving_variance
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_5AssignVariableOp$depthwise_conv2d_10/depthwise_kernel
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_6AssignVariableOpbatch_normalization_49/gamma
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_7AssignVariableOpbatch_normalization_49/beta
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_8AssignVariableOp"batch_normalization_49/moving_mean
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
w
AssignVariableOp_9AssignVariableOp&batch_normalization_49/moving_varianceIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
v
AssignVariableOp_10AssignVariableOp$separable_conv2d_10/depthwise_kernelIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
v
AssignVariableOp_11AssignVariableOp$separable_conv2d_10/pointwise_kernelIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_12AssignVariableOpbatch_normalization_50/gammaIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_13AssignVariableOpbatch_normalization_50/betaIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_14AssignVariableOp"batch_normalization_50/moving_meanIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_15AssignVariableOp&batch_normalization_50/moving_varianceIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_16AssignVariableOpdense/kernelIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_17AssignVariableOp
dense/biasIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0	*
_output_shapes
:
Z
AssignVariableOp_18AssignVariableOpSGD/iterIdentity_19"/device:CPU:0*
dtype0	
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_19AssignVariableOp	SGD/decayIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_20AssignVariableOpSGD/momentumIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_21AssignVariableOptotalIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_22AssignVariableOpcountIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_23AssignVariableOptotal_1Identity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_24AssignVariableOpcount_1Identity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
o
AssignVariableOp_25AssignVariableOpSGD/conv2d_28/kernel/momentumIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
{
AssignVariableOp_26AssignVariableOp)SGD/batch_normalization_48/gamma/momentumIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
z
AssignVariableOp_27AssignVariableOp(SGD/batch_normalization_48/beta/momentumIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
?
AssignVariableOp_28AssignVariableOp1SGD/depthwise_conv2d_10/depthwise_kernel/momentumIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
{
AssignVariableOp_29AssignVariableOp)SGD/batch_normalization_49/gamma/momentumIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
z
AssignVariableOp_30AssignVariableOp(SGD/batch_normalization_49/beta/momentumIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
?
AssignVariableOp_31AssignVariableOp1SGD/separable_conv2d_10/depthwise_kernel/momentumIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
?
AssignVariableOp_32AssignVariableOp1SGD/separable_conv2d_10/pointwise_kernel/momentumIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
{
AssignVariableOp_33AssignVariableOp)SGD/batch_normalization_50/gamma/momentumIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
z
AssignVariableOp_34AssignVariableOp(SGD/batch_normalization_50/beta/momentumIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_35AssignVariableOpSGD/dense/kernel/momentumIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_36AssignVariableOpSGD/dense/bias/momentumIdentity_37"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_38Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??

??
?
F__inference_model_16_layer_call_and_return_conditional_losses_11055520
input_11,
(conv2d_28_conv2d_readvariableop_resource2
.batch_normalization_48_readvariableop_resource4
0batch_normalization_48_readvariableop_1_resourceC
?batch_normalization_48_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource9
5depthwise_conv2d_10_depthwise_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource@
<separable_conv2d_10_separable_conv2d_readvariableop_resourceB
>separable_conv2d_10_separable_conv2d_readvariableop_1_resource2
.batch_normalization_50_readvariableop_resource4
0batch_normalization_50_readvariableop_1_resourceC
?batch_normalization_50_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??%batch_normalization_48/AssignNewValue?'batch_normalization_48/AssignNewValue_1?6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1?%batch_normalization_49/AssignNewValue?'batch_normalization_49/AssignNewValue_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?%batch_normalization_50/AssignNewValue?'batch_normalization_50/AssignNewValue_1?6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_50/ReadVariableOp?'batch_normalization_50/ReadVariableOp_1?conv2d_28/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?,depthwise_conv2d_10/depthwise/ReadVariableOp?3separable_conv2d_10/separable_conv2d/ReadVariableOp?5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinput_11'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
conv2d_28/Conv2D?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_48/ReadVariableOp?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_48/ReadVariableOp_1?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_28/Conv2D:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_48/FusedBatchNormV3?
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_48/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_48/AssignNewValue?
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_48/AssignNewValue_1?
,depthwise_conv2d_10/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_10_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_10/depthwise/ReadVariableOp?
#depthwise_conv2d_10/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_10/depthwise/Shape?
+depthwise_conv2d_10/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_10/depthwise/dilation_rate?
depthwise_conv2d_10/depthwiseDepthwiseConv2dNative+batch_normalization_48/FusedBatchNormV3:y:04depthwise_conv2d_10/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
depthwise_conv2d_10/depthwise?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_10/depthwise:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_49/FusedBatchNormV3?
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_49/AssignNewValue?
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_49/AssignNewValue_1?
activation_20/EluElu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????d2
activation_20/Elu?
average_pooling2d_33/AvgPoolAvgPoolactivation_20/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_33/AvgPooly
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_26/dropout/Const?
dropout_26/dropout/MulMul%average_pooling2d_33/AvgPool:output:0!dropout_26/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_26/dropout/Mul?
dropout_26/dropout/ShapeShape%average_pooling2d_33/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shape?
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform?
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_26/dropout/GreaterEqual/y?
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_26/dropout/GreaterEqual?
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_26/dropout/Cast?
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_26/dropout/Mul_1?
3separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_10_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3separable_conv2d_10/separable_conv2d/ReadVariableOp?
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_10_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_10/separable_conv2d/Shape?
2separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_10/separable_conv2d/dilation_rate?
.separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_26/dropout/Mul_1:z:0;separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
20
.separable_conv2d_10/separable_conv2d/depthwise?
$separable_conv2d_10/separable_conv2dConv2D7separable_conv2d_10/separable_conv2d/depthwise:output:0=separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2&
$separable_conv2d_10/separable_conv2d?
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_50/ReadVariableOp?
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_50/ReadVariableOp_1?
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_10/separable_conv2d:output:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_50/FusedBatchNormV3?
%batch_normalization_50/AssignNewValueAssignVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource4batch_normalization_50/FusedBatchNormV3:batch_mean:07^batch_normalization_50/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_50/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_50/AssignNewValue?
'batch_normalization_50/AssignNewValue_1AssignVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_50/FusedBatchNormV3:batch_variance:09^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_50/AssignNewValue_1?
activation_21/EluElu+batch_normalization_50/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation_21/Elu?
average_pooling2d_34/AvgPoolAvgPoolactivation_21/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_34/AvgPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_27/dropout/Const?
dropout_27/dropout/MulMul%average_pooling2d_34/AvgPool:output:0!dropout_27/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_27/dropout/Mul?
dropout_27/dropout/ShapeShape%average_pooling2d_34/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape?
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform?
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_27/dropout/GreaterEqual/y?
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_27/dropout/GreaterEqual?
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_27/dropout/Cast?
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_27/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
flatten/Const?
flatten/ReshapeReshapedropout_27/dropout/Mul_1:z:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????02
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddw
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?	
IdentityIdentitysoftmax/Softmax:softmax:0&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1&^batch_normalization_50/AssignNewValue(^batch_normalization_50/AssignNewValue_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1 ^conv2d_28/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^depthwise_conv2d_10/depthwise/ReadVariableOp4^separable_conv2d_10/separable_conv2d/ReadVariableOp6^separable_conv2d_10/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????d::::::::::::::::::2N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_50/AssignNewValue%batch_normalization_50/AssignNewValue2R
'batch_normalization_50/AssignNewValue_1'batch_normalization_50/AssignNewValue_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,depthwise_conv2d_10/depthwise/ReadVariableOp,depthwise_conv2d_10/depthwise/ReadVariableOp2j
3separable_conv2d_10/separable_conv2d/ReadVariableOp3separable_conv2d_10/separable_conv2d/ReadVariableOp2n
5separable_conv2d_10/separable_conv2d/ReadVariableOp_15separable_conv2d_10/separable_conv2d/ReadVariableOp_1:Y U
/
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055853

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_11056159

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
7__inference_average_pooling2d_34_layer_call_fn_11055034

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_28_layer_call_and_return_conditional_losses_11055826

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????d:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
Q__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_11054931

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0 ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_11056399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_48_layer_call_fn_11055985

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_11056164

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_48_layer_call_fn_11055891

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_48_layer_call_fn_11055967

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
a
E__inference_softmax_layer_call_and_return_conditional_losses_11056414

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
7__inference_average_pooling2d_33_layer_call_fn_11054918

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
+__inference_model_16_layer_call_fn_11055695
input_11,
(conv2d_28_conv2d_readvariableop_resource2
.batch_normalization_48_readvariableop_resource4
0batch_normalization_48_readvariableop_1_resourceC
?batch_normalization_48_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource9
5depthwise_conv2d_10_depthwise_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource@
<separable_conv2d_10_separable_conv2d_readvariableop_resourceB
>separable_conv2d_10_separable_conv2d_readvariableop_1_resource2
.batch_normalization_50_readvariableop_resource4
0batch_normalization_50_readvariableop_1_resourceC
?batch_normalization_50_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??%batch_normalization_48/AssignNewValue?'batch_normalization_48/AssignNewValue_1?6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1?%batch_normalization_49/AssignNewValue?'batch_normalization_49/AssignNewValue_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?%batch_normalization_50/AssignNewValue?'batch_normalization_50/AssignNewValue_1?6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_50/ReadVariableOp?'batch_normalization_50/ReadVariableOp_1?conv2d_28/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?,depthwise_conv2d_10/depthwise/ReadVariableOp?3separable_conv2d_10/separable_conv2d/ReadVariableOp?5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinput_11'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
conv2d_28/Conv2D?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_48/ReadVariableOp?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_48/ReadVariableOp_1?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_28/Conv2D:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_48/FusedBatchNormV3?
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_48/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_48/AssignNewValue?
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_48/AssignNewValue_1?
,depthwise_conv2d_10/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_10_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_10/depthwise/ReadVariableOp?
#depthwise_conv2d_10/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_10/depthwise/Shape?
+depthwise_conv2d_10/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_10/depthwise/dilation_rate?
depthwise_conv2d_10/depthwiseDepthwiseConv2dNative+batch_normalization_48/FusedBatchNormV3:y:04depthwise_conv2d_10/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
depthwise_conv2d_10/depthwise?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_10/depthwise:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_49/FusedBatchNormV3?
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_49/AssignNewValue?
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_49/AssignNewValue_1?
activation_20/EluElu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????d2
activation_20/Elu?
average_pooling2d_33/AvgPoolAvgPoolactivation_20/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_33/AvgPooly
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_26/dropout/Const?
dropout_26/dropout/MulMul%average_pooling2d_33/AvgPool:output:0!dropout_26/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_26/dropout/Mul?
dropout_26/dropout/ShapeShape%average_pooling2d_33/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shape?
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform?
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_26/dropout/GreaterEqual/y?
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_26/dropout/GreaterEqual?
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_26/dropout/Cast?
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_26/dropout/Mul_1?
3separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_10_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3separable_conv2d_10/separable_conv2d/ReadVariableOp?
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_10_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_10/separable_conv2d/Shape?
2separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_10/separable_conv2d/dilation_rate?
.separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_26/dropout/Mul_1:z:0;separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
20
.separable_conv2d_10/separable_conv2d/depthwise?
$separable_conv2d_10/separable_conv2dConv2D7separable_conv2d_10/separable_conv2d/depthwise:output:0=separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2&
$separable_conv2d_10/separable_conv2d?
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_50/ReadVariableOp?
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_50/ReadVariableOp_1?
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_10/separable_conv2d:output:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_50/FusedBatchNormV3?
%batch_normalization_50/AssignNewValueAssignVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource4batch_normalization_50/FusedBatchNormV3:batch_mean:07^batch_normalization_50/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_50/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_50/AssignNewValue?
'batch_normalization_50/AssignNewValue_1AssignVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_50/FusedBatchNormV3:batch_variance:09^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_50/AssignNewValue_1?
activation_21/EluElu+batch_normalization_50/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation_21/Elu?
average_pooling2d_34/AvgPoolAvgPoolactivation_21/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_34/AvgPooly
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_27/dropout/Const?
dropout_27/dropout/MulMul%average_pooling2d_34/AvgPool:output:0!dropout_27/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout_27/dropout/Mul?
dropout_27/dropout/ShapeShape%average_pooling2d_34/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape?
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform?
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_27/dropout/GreaterEqual/y?
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2!
dropout_27/dropout/GreaterEqual?
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout_27/dropout/Cast?
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout_27/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
flatten/Const?
flatten/ReshapeReshapedropout_27/dropout/Mul_1:z:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????02
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddw
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?	
IdentityIdentitysoftmax/Softmax:softmax:0&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1&^batch_normalization_50/AssignNewValue(^batch_normalization_50/AssignNewValue_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1 ^conv2d_28/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^depthwise_conv2d_10/depthwise/ReadVariableOp4^separable_conv2d_10/separable_conv2d/ReadVariableOp6^separable_conv2d_10/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????d::::::::::::::::::2N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_50/AssignNewValue%batch_normalization_50/AssignNewValue2R
'batch_normalization_50/AssignNewValue_1'batch_normalization_50/AssignNewValue_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,depthwise_conv2d_10/depthwise/ReadVariableOp,depthwise_conv2d_10/depthwise/ReadVariableOp2j
3separable_conv2d_10/separable_conv2d/ReadVariableOp3separable_conv2d_10/separable_conv2d/ReadVariableOp2n
5separable_conv2d_10/separable_conv2d/ReadVariableOp_15separable_conv2d_10/separable_conv2d/ReadVariableOp_1:Y U
/
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
?
9__inference_batch_normalization_48_layer_call_fn_11055909

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_49_layer_call_fn_11056061

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056023

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_27_layer_call_and_return_conditional_losses_11056360

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_11056383

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????02	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
-__inference_dropout_27_layer_call_fn_11056372

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_softmax_layer_call_fn_11056419

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055871

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?k
?
+__inference_model_16_layer_call_fn_11055772
input_11,
(conv2d_28_conv2d_readvariableop_resource2
.batch_normalization_48_readvariableop_resource4
0batch_normalization_48_readvariableop_1_resourceC
?batch_normalization_48_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource9
5depthwise_conv2d_10_depthwise_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource@
<separable_conv2d_10_separable_conv2d_readvariableop_resourceB
>separable_conv2d_10_separable_conv2d_readvariableop_1_resource2
.batch_normalization_50_readvariableop_resource4
0batch_normalization_50_readvariableop_1_resourceC
?batch_normalization_50_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_50/ReadVariableOp?'batch_normalization_50/ReadVariableOp_1?conv2d_28/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?,depthwise_conv2d_10/depthwise/ReadVariableOp?3separable_conv2d_10/separable_conv2d/ReadVariableOp?5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinput_11'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
conv2d_28/Conv2D?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_48/ReadVariableOp?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_48/ReadVariableOp_1?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_28/Conv2D:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_48/FusedBatchNormV3?
,depthwise_conv2d_10/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_10_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_10/depthwise/ReadVariableOp?
#depthwise_conv2d_10/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_10/depthwise/Shape?
+depthwise_conv2d_10/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_10/depthwise/dilation_rate?
depthwise_conv2d_10/depthwiseDepthwiseConv2dNative+batch_normalization_48/FusedBatchNormV3:y:04depthwise_conv2d_10/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
depthwise_conv2d_10/depthwise?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_10/depthwise:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_49/FusedBatchNormV3?
activation_20/EluElu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????d2
activation_20/Elu?
average_pooling2d_33/AvgPoolAvgPoolactivation_20/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_33/AvgPool?
dropout_26/IdentityIdentity%average_pooling2d_33/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_26/Identity?
3separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_10_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3separable_conv2d_10/separable_conv2d/ReadVariableOp?
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_10_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_10/separable_conv2d/Shape?
2separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_10/separable_conv2d/dilation_rate?
.separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_26/Identity:output:0;separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
20
.separable_conv2d_10/separable_conv2d/depthwise?
$separable_conv2d_10/separable_conv2dConv2D7separable_conv2d_10/separable_conv2d/depthwise:output:0=separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2&
$separable_conv2d_10/separable_conv2d?
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_50/ReadVariableOp?
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_50/ReadVariableOp_1?
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_10/separable_conv2d:output:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_50/FusedBatchNormV3?
activation_21/EluElu+batch_normalization_50/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation_21/Elu?
average_pooling2d_34/AvgPoolAvgPoolactivation_21/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_34/AvgPool?
dropout_27/IdentityIdentity%average_pooling2d_34/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_27/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
flatten/Const?
flatten/ReshapeReshapedropout_27/Identity:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????02
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddw
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1 ^conv2d_28/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^depthwise_conv2d_10/depthwise/ReadVariableOp4^separable_conv2d_10/separable_conv2d/ReadVariableOp6^separable_conv2d_10/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????d::::::::::::::::::2p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,depthwise_conv2d_10/depthwise/ReadVariableOp,depthwise_conv2d_10/depthwise/ReadVariableOp2j
3separable_conv2d_10/separable_conv2d/ReadVariableOp3separable_conv2d_10/separable_conv2d/ReadVariableOp2n
5separable_conv2d_10/separable_conv2d/ReadVariableOp_15separable_conv2d_10/separable_conv2d/ReadVariableOp_1:Y U
/
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056005

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_49_layer_call_fn_11056043

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055929

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
g
K__inference_activation_21_layer_call_and_return_conditional_losses_11056338

inputs
identityS
EluEluinputs*
T0*/
_output_shapes
:?????????2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_28_layer_call_fn_11055833

inputs"
conv2d_readvariableop_resource
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
Conv2D?
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????d:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056277

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055947

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056081

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
Q__inference_depthwise_conv2d_10_layer_call_and_return_conditional_losses_11054818

inputs%
!depthwise_readvariableop_resource
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
n
R__inference_average_pooling2d_34_layer_call_and_return_conditional_losses_11055028

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?k
?
F__inference_model_16_layer_call_and_return_conditional_losses_11055597
input_11,
(conv2d_28_conv2d_readvariableop_resource2
.batch_normalization_48_readvariableop_resource4
0batch_normalization_48_readvariableop_1_resourceC
?batch_normalization_48_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource9
5depthwise_conv2d_10_depthwise_readvariableop_resource2
.batch_normalization_49_readvariableop_resource4
0batch_normalization_49_readvariableop_1_resourceC
?batch_normalization_49_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource@
<separable_conv2d_10_separable_conv2d_readvariableop_resourceB
>separable_conv2d_10_separable_conv2d_readvariableop_1_resource2
.batch_normalization_50_readvariableop_resource4
0batch_normalization_50_readvariableop_1_resourceC
?batch_normalization_50_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_48/ReadVariableOp?'batch_normalization_48/ReadVariableOp_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_50/ReadVariableOp?'batch_normalization_50/ReadVariableOp_1?conv2d_28/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?,depthwise_conv2d_10/depthwise/ReadVariableOp?3separable_conv2d_10/separable_conv2d/ReadVariableOp?5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dinput_11'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
conv2d_28/Conv2D?
%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_48/ReadVariableOp?
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_48/ReadVariableOp_1?
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3conv2d_28/Conv2D:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_48/FusedBatchNormV3?
,depthwise_conv2d_10/depthwise/ReadVariableOpReadVariableOp5depthwise_conv2d_10_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02.
,depthwise_conv2d_10/depthwise/ReadVariableOp?
#depthwise_conv2d_10/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2%
#depthwise_conv2d_10/depthwise/Shape?
+depthwise_conv2d_10/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2-
+depthwise_conv2d_10/depthwise/dilation_rate?
depthwise_conv2d_10/depthwiseDepthwiseConv2dNative+batch_normalization_48/FusedBatchNormV3:y:04depthwise_conv2d_10/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
depthwise_conv2d_10/depthwise?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_49/ReadVariableOp?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_49/ReadVariableOp_1?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3&depthwise_conv2d_10/depthwise:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_49/FusedBatchNormV3?
activation_20/EluElu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????d2
activation_20/Elu?
average_pooling2d_33/AvgPoolAvgPoolactivation_20/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_33/AvgPool?
dropout_26/IdentityIdentity%average_pooling2d_33/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_26/Identity?
3separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOp<separable_conv2d_10_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3separable_conv2d_10/separable_conv2d/ReadVariableOp?
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOp>separable_conv2d_10_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype027
5separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
*separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2,
*separable_conv2d_10/separable_conv2d/Shape?
2separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      24
2separable_conv2d_10/separable_conv2d/dilation_rate?
.separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_26/Identity:output:0;separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
20
.separable_conv2d_10/separable_conv2d/depthwise?
$separable_conv2d_10/separable_conv2dConv2D7separable_conv2d_10/separable_conv2d/depthwise:output:0=separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2&
$separable_conv2d_10/separable_conv2d?
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_50/ReadVariableOp?
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_50/ReadVariableOp_1?
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3-separable_conv2d_10/separable_conv2d:output:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_50/FusedBatchNormV3?
activation_21/EluElu+batch_normalization_50/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation_21/Elu?
average_pooling2d_34/AvgPoolAvgPoolactivation_21/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling2d_34/AvgPool?
dropout_27/IdentityIdentity%average_pooling2d_34/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
dropout_27/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
flatten/Const?
flatten/ReshapeReshapedropout_27/Identity:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????02
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddw
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1 ^conv2d_28/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^depthwise_conv2d_10/depthwise/ReadVariableOp4^separable_conv2d_10/separable_conv2d/ReadVariableOp6^separable_conv2d_10/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????d::::::::::::::::::2p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2\
,depthwise_conv2d_10/depthwise/ReadVariableOp,depthwise_conv2d_10/depthwise/ReadVariableOp2j
3separable_conv2d_10/separable_conv2d/ReadVariableOp3separable_conv2d_10/separable_conv2d/ReadVariableOp2n
5separable_conv2d_10/separable_conv2d/ReadVariableOp_15separable_conv2d_10/separable_conv2d/ReadVariableOp_1:Y U
/
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
?
9__inference_batch_normalization_49_layer_call_fn_11056119

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_49_layer_call_fn_11056137

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056099

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????d::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_50_layer_call_fn_11056333

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_50_layer_call_fn_11056315

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056295

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_activation_20_layer_call_fn_11056147

inputs
identityS
EluEluinputs*
T0*/
_output_shapes
:?????????d2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_50_layer_call_fn_11056239

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
6__inference_depthwise_conv2d_10_layer_call_fn_11054828

inputs%
!depthwise_readvariableop_resource
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape?
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
	depthwise?
IdentityIdentitydepthwise:output:0^depthwise/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????:24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_activation_21_layer_call_fn_11056343

inputs
identityS
EluEluinputs*
T0*/
_output_shapes
:?????????2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
&__inference_signature_wrapper_11055819
input_11
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_110547302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????d::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
F
*__inference_flatten_layer_call_fn_11056389

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????02	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
-__inference_dropout_27_layer_call_fn_11056377

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_11056409

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
6__inference_separable_conv2d_10_layer_call_fn_11054944

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource
identity??separable_conv2d/ReadVariableOp?!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
separable_conv2d/ReadVariableOp?
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02#
!separable_conv2d/ReadVariableOp_1?
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
separable_conv2d/Shape?
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2 
separable_conv2d/dilation_rate?
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
separable_conv2d/depthwise?
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
separable_conv2d?
IdentityIdentityseparable_conv2d:output:0 ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
n
R__inference_average_pooling2d_33_layer_call_and_return_conditional_losses_11054912

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_activation_20_layer_call_and_return_conditional_losses_11056142

inputs
identityS
EluEluinputs*
T0*/
_output_shapes
:?????????d2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
g
H__inference_dropout_27_layer_call_and_return_conditional_losses_11056355

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
-__inference_dropout_26_layer_call_fn_11056176

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?{
?
#__inference__wrapped_model_11054730
input_115
1model_16_conv2d_28_conv2d_readvariableop_resource;
7model_16_batch_normalization_48_readvariableop_resource=
9model_16_batch_normalization_48_readvariableop_1_resourceL
Hmodel_16_batch_normalization_48_fusedbatchnormv3_readvariableop_resourceN
Jmodel_16_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resourceB
>model_16_depthwise_conv2d_10_depthwise_readvariableop_resource;
7model_16_batch_normalization_49_readvariableop_resource=
9model_16_batch_normalization_49_readvariableop_1_resourceL
Hmodel_16_batch_normalization_49_fusedbatchnormv3_readvariableop_resourceN
Jmodel_16_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resourceI
Emodel_16_separable_conv2d_10_separable_conv2d_readvariableop_resourceK
Gmodel_16_separable_conv2d_10_separable_conv2d_readvariableop_1_resource;
7model_16_batch_normalization_50_readvariableop_resource=
9model_16_batch_normalization_50_readvariableop_1_resourceL
Hmodel_16_batch_normalization_50_fusedbatchnormv3_readvariableop_resourceN
Jmodel_16_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource1
-model_16_dense_matmul_readvariableop_resource2
.model_16_dense_biasadd_readvariableop_resource
identity???model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp?Amodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?.model_16/batch_normalization_48/ReadVariableOp?0model_16/batch_normalization_48/ReadVariableOp_1??model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?Amodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?.model_16/batch_normalization_49/ReadVariableOp?0model_16/batch_normalization_49/ReadVariableOp_1??model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp?Amodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?.model_16/batch_normalization_50/ReadVariableOp?0model_16/batch_normalization_50/ReadVariableOp_1?(model_16/conv2d_28/Conv2D/ReadVariableOp?%model_16/dense/BiasAdd/ReadVariableOp?$model_16/dense/MatMul/ReadVariableOp?5model_16/depthwise_conv2d_10/depthwise/ReadVariableOp?<model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp?>model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
(model_16/conv2d_28/Conv2D/ReadVariableOpReadVariableOp1model_16_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_16/conv2d_28/Conv2D/ReadVariableOp?
model_16/conv2d_28/Conv2DConv2Dinput_110model_16/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
2
model_16/conv2d_28/Conv2D?
.model_16/batch_normalization_48/ReadVariableOpReadVariableOp7model_16_batch_normalization_48_readvariableop_resource*
_output_shapes
:*
dtype020
.model_16/batch_normalization_48/ReadVariableOp?
0model_16/batch_normalization_48/ReadVariableOp_1ReadVariableOp9model_16_batch_normalization_48_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_16/batch_normalization_48/ReadVariableOp_1?
?model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_16_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp?
Amodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_16_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?
0model_16/batch_normalization_48/FusedBatchNormV3FusedBatchNormV3"model_16/conv2d_28/Conv2D:output:06model_16/batch_normalization_48/ReadVariableOp:value:08model_16/batch_normalization_48/ReadVariableOp_1:value:0Gmodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0Imodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 22
0model_16/batch_normalization_48/FusedBatchNormV3?
5model_16/depthwise_conv2d_10/depthwise/ReadVariableOpReadVariableOp>model_16_depthwise_conv2d_10_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype027
5model_16/depthwise_conv2d_10/depthwise/ReadVariableOp?
,model_16/depthwise_conv2d_10/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2.
,model_16/depthwise_conv2d_10/depthwise/Shape?
4model_16/depthwise_conv2d_10/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      26
4model_16/depthwise_conv2d_10/depthwise/dilation_rate?
&model_16/depthwise_conv2d_10/depthwiseDepthwiseConv2dNative4model_16/batch_normalization_48/FusedBatchNormV3:y:0=model_16/depthwise_conv2d_10/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2(
&model_16/depthwise_conv2d_10/depthwise?
.model_16/batch_normalization_49/ReadVariableOpReadVariableOp7model_16_batch_normalization_49_readvariableop_resource*
_output_shapes
:*
dtype020
.model_16/batch_normalization_49/ReadVariableOp?
0model_16/batch_normalization_49/ReadVariableOp_1ReadVariableOp9model_16_batch_normalization_49_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_16/batch_normalization_49/ReadVariableOp_1?
?model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_16_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?
Amodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_16_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?
0model_16/batch_normalization_49/FusedBatchNormV3FusedBatchNormV3/model_16/depthwise_conv2d_10/depthwise:output:06model_16/batch_normalization_49/ReadVariableOp:value:08model_16/batch_normalization_49/ReadVariableOp_1:value:0Gmodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0Imodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:::::*
epsilon%o?:*
is_training( 22
0model_16/batch_normalization_49/FusedBatchNormV3?
model_16/activation_20/EluElu4model_16/batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????d2
model_16/activation_20/Elu?
%model_16/average_pooling2d_33/AvgPoolAvgPool(model_16/activation_20/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%model_16/average_pooling2d_33/AvgPool?
model_16/dropout_26/IdentityIdentity.model_16/average_pooling2d_33/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
model_16/dropout_26/Identity?
<model_16/separable_conv2d_10/separable_conv2d/ReadVariableOpReadVariableOpEmodel_16_separable_conv2d_10_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp?
>model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1ReadVariableOpGmodel_16_separable_conv2d_10_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype02@
>model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1?
3model_16/separable_conv2d_10/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            25
3model_16/separable_conv2d_10/separable_conv2d/Shape?
;model_16/separable_conv2d_10/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_16/separable_conv2d_10/separable_conv2d/dilation_rate?
7model_16/separable_conv2d_10/separable_conv2d/depthwiseDepthwiseConv2dNative%model_16/dropout_26/Identity:output:0Dmodel_16/separable_conv2d_10/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
29
7model_16/separable_conv2d_10/separable_conv2d/depthwise?
-model_16/separable_conv2d_10/separable_conv2dConv2D@model_16/separable_conv2d_10/separable_conv2d/depthwise:output:0Fmodel_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2/
-model_16/separable_conv2d_10/separable_conv2d?
.model_16/batch_normalization_50/ReadVariableOpReadVariableOp7model_16_batch_normalization_50_readvariableop_resource*
_output_shapes
:*
dtype020
.model_16/batch_normalization_50/ReadVariableOp?
0model_16/batch_normalization_50/ReadVariableOp_1ReadVariableOp9model_16_batch_normalization_50_readvariableop_1_resource*
_output_shapes
:*
dtype022
0model_16/batch_normalization_50/ReadVariableOp_1?
?model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_16_batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp?
Amodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_16_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Amodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?
0model_16/batch_normalization_50/FusedBatchNormV3FusedBatchNormV36model_16/separable_conv2d_10/separable_conv2d:output:06model_16/batch_normalization_50/ReadVariableOp:value:08model_16/batch_normalization_50/ReadVariableOp_1:value:0Gmodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0Imodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 22
0model_16/batch_normalization_50/FusedBatchNormV3?
model_16/activation_21/EluElu4model_16/batch_normalization_50/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
model_16/activation_21/Elu?
%model_16/average_pooling2d_34/AvgPoolAvgPool(model_16/activation_21/Elu:activations:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%model_16/average_pooling2d_34/AvgPool?
model_16/dropout_27/IdentityIdentity.model_16/average_pooling2d_34/AvgPool:output:0*
T0*/
_output_shapes
:?????????2
model_16/dropout_27/Identity?
model_16/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   2
model_16/flatten/Const?
model_16/flatten/ReshapeReshape%model_16/dropout_27/Identity:output:0model_16/flatten/Const:output:0*
T0*'
_output_shapes
:?????????02
model_16/flatten/Reshape?
$model_16/dense/MatMul/ReadVariableOpReadVariableOp-model_16_dense_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02&
$model_16/dense/MatMul/ReadVariableOp?
model_16/dense/MatMulMatMul!model_16/flatten/Reshape:output:0,model_16/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_16/dense/MatMul?
%model_16/dense/BiasAdd/ReadVariableOpReadVariableOp.model_16_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_16/dense/BiasAdd/ReadVariableOp?
model_16/dense/BiasAddBiasAddmodel_16/dense/MatMul:product:0-model_16/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_16/dense/BiasAdd?
model_16/softmax/SoftmaxSoftmaxmodel_16/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_16/softmax/Softmax?
IdentityIdentity"model_16/softmax/Softmax:softmax:0@^model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOpB^model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1/^model_16/batch_normalization_48/ReadVariableOp1^model_16/batch_normalization_48/ReadVariableOp_1@^model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOpB^model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1/^model_16/batch_normalization_49/ReadVariableOp1^model_16/batch_normalization_49/ReadVariableOp_1@^model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOpB^model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1/^model_16/batch_normalization_50/ReadVariableOp1^model_16/batch_normalization_50/ReadVariableOp_1)^model_16/conv2d_28/Conv2D/ReadVariableOp&^model_16/dense/BiasAdd/ReadVariableOp%^model_16/dense/MatMul/ReadVariableOp6^model_16/depthwise_conv2d_10/depthwise/ReadVariableOp=^model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp?^model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????d::::::::::::::::::2?
?model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp?model_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2?
Amodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Amodel_16/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12`
.model_16/batch_normalization_48/ReadVariableOp.model_16/batch_normalization_48/ReadVariableOp2d
0model_16/batch_normalization_48/ReadVariableOp_10model_16/batch_normalization_48/ReadVariableOp_12?
?model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?model_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2?
Amodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Amodel_16/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12`
.model_16/batch_normalization_49/ReadVariableOp.model_16/batch_normalization_49/ReadVariableOp2d
0model_16/batch_normalization_49/ReadVariableOp_10model_16/batch_normalization_49/ReadVariableOp_12?
?model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp?model_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp2?
Amodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Amodel_16/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12`
.model_16/batch_normalization_50/ReadVariableOp.model_16/batch_normalization_50/ReadVariableOp2d
0model_16/batch_normalization_50/ReadVariableOp_10model_16/batch_normalization_50/ReadVariableOp_12T
(model_16/conv2d_28/Conv2D/ReadVariableOp(model_16/conv2d_28/Conv2D/ReadVariableOp2N
%model_16/dense/BiasAdd/ReadVariableOp%model_16/dense/BiasAdd/ReadVariableOp2L
$model_16/dense/MatMul/ReadVariableOp$model_16/dense/MatMul/ReadVariableOp2n
5model_16/depthwise_conv2d_10/depthwise/ReadVariableOp5model_16/depthwise_conv2d_10/depthwise/ReadVariableOp2|
<model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp<model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp2?
>model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1>model_16/separable_conv2d_10/separable_conv2d/ReadVariableOp_1:Y U
/
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
?
9__inference_batch_normalization_50_layer_call_fn_11056257

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
-__inference_dropout_26_layer_call_fn_11056181

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?-
saver_filename:0
Identity:0Identity_388"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_119
serving_default_input_11:0?????????d;
softmax0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
τ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?
_tf_keras_network?{"class_name": "Functional", "name": "model_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 20]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_48", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_48", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_10", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [6, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 2, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}}, "name": "depthwise_conv2d_10", "inbound_nodes": [[["batch_normalization_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_49", "inbound_nodes": [[["depthwise_conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "elu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_49", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 4]}, "data_format": "channels_last"}, "name": "average_pooling2d_33", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_26", "inbound_nodes": [[["average_pooling2d_33", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_10", "inbound_nodes": [[["dropout_26", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_50", "inbound_nodes": [[["separable_conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "elu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_50", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 8]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 8]}, "data_format": "channels_last"}, "name": "average_pooling2d_34", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_27", "inbound_nodes": [[["average_pooling2d_34", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["dropout_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 0.25, "axis": 0}}, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "softmax", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["softmax", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 100, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 20]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_48", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_48", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_10", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [6, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 2, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}}, "name": "depthwise_conv2d_10", "inbound_nodes": [[["batch_normalization_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_49", "inbound_nodes": [[["depthwise_conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "elu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_49", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 4]}, "data_format": "channels_last"}, "name": "average_pooling2d_33", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_26", "inbound_nodes": [[["average_pooling2d_33", 0, 0, {}]]]}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "name": "separable_conv2d_10", "inbound_nodes": [[["dropout_26", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_50", "inbound_nodes": [[["separable_conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "elu"}, "name": "activation_21", "inbound_nodes": [[["batch_normalization_50", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 8]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 8]}, "data_format": "channels_last"}, "name": "average_pooling2d_34", "inbound_nodes": [[["activation_21", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_27", "inbound_nodes": [[["average_pooling2d_34", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["dropout_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 0.25, "axis": 0}}, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "softmax", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["softmax", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": {"class_name": "ExponentialDecay", "config": {"initial_learning_rate": 0.01, "decay_steps": 500, "decay_rate": 0.9, "staircase": false, "name": null}}, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": true}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}}
?


kernel
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 100, 1]}, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 20]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 100, 1]}}
?	
axis
	gamma
beta
moving_mean
 moving_variance
!	variables
"regularization_losses
#trainable_variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_48", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_48", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 100, 8]}}
?

%depthwise_kernel
&	variables
'regularization_losses
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_10", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [6, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 2, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": {"class_name": "MaxNorm", "config": {"max_value": 1.0, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 100, 8]}}
?	
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/	variables
0regularization_losses
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_49", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_49", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 100, 16]}}
?
3	variables
4regularization_losses
5trainable_variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "elu"}}
?
7	variables
8regularization_losses
9trainable_variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling2D", "name": "average_pooling2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
;	variables
<regularization_losses
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
?depthwise_kernel
@pointwise_kernel
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SeparableConv2D", "name": "separable_conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "separable_conv2d_10", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [1, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 25, 16]}}
?	
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_50", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_50", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 25, 16]}}
?
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "elu"}}
?
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling2D", "name": "average_pooling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 8]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 8]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_27", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 0.25, "axis": 0}}, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48]}}
?
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}}
?
hiter
	idecay
jmomentummomentum?momentum?momentum?%momentum?+momentum?,momentum??momentum?@momentum?Fmomentum?Gmomentum?^momentum?_momentum?"
	optimizer
?
0
1
2
3
 4
%5
+6
,7
-8
.9
?10
@11
F12
G13
H14
I15
^16
_17"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
%3
+4
,5
?6
@7
F8
G9
^10
_11"
trackable_list_wrapper
?
	variables
klayer_metrics
lmetrics
mnon_trainable_variables

nlayers
regularization_losses
trainable_variables
olayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_28/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
	variables
player_metrics
qmetrics
rnon_trainable_variables

slayers
regularization_losses
trainable_variables
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_48/gamma
):'2batch_normalization_48/beta
2:0 (2"batch_normalization_48/moving_mean
6:4 (2&batch_normalization_48/moving_variance
<
0
1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
!	variables
ulayer_metrics
vmetrics
wnon_trainable_variables

xlayers
"regularization_losses
#trainable_variables
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<2$depthwise_conv2d_10/depthwise_kernel
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
?
&	variables
zlayer_metrics
{metrics
|non_trainable_variables

}layers
'regularization_losses
(trainable_variables
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_49/gamma
):'2batch_normalization_49/beta
2:0 (2"batch_normalization_49/moving_mean
6:4 (2&batch_normalization_49/moving_variance
<
+0
,1
-2
.3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
/	variables
layer_metrics
?metrics
?non_trainable_variables
?layers
0regularization_losses
1trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
3	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
4regularization_losses
5trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
8regularization_losses
9trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
<regularization_losses
=trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<2$separable_conv2d_10/depthwise_kernel
>:<2$separable_conv2d_10/pointwise_kernel
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
A	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Bregularization_losses
Ctrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_50/gamma
):'2batch_normalization_50/beta
2:0 (2"batch_normalization_50/moving_mean
6:4 (2&batch_normalization_50/moving_variance
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
J	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Kregularization_losses
Ltrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
N	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Oregularization_losses
Ptrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
R	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Sregularization_losses
Ttrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
V	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
Wregularization_losses
Xtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Z	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
[regularization_losses
\trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:02dense/kernel
:2
dense/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
`	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
aregularization_losses
btrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
d	variables
?layer_metrics
?metrics
?non_trainable_variables
?layers
eregularization_losses
ftrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/momentum
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
J
0
 1
-2
.3
H4
I5"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
5:32SGD/conv2d_28/kernel/momentum
5:32)SGD/batch_normalization_48/gamma/momentum
4:22(SGD/batch_normalization_48/beta/momentum
I:G21SGD/depthwise_conv2d_10/depthwise_kernel/momentum
5:32)SGD/batch_normalization_49/gamma/momentum
4:22(SGD/batch_normalization_49/beta/momentum
I:G21SGD/separable_conv2d_10/depthwise_kernel/momentum
I:G21SGD/separable_conv2d_10/pointwise_kernel/momentum
5:32)SGD/batch_normalization_50/gamma/momentum
4:22(SGD/batch_normalization_50/beta/momentum
):'02SGD/dense/kernel/momentum
#:!2SGD/dense/bias/momentum
?2?
+__inference_model_16_layer_call_fn_11055695
+__inference_model_16_layer_call_fn_11055772?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_16_layer_call_and_return_conditional_losses_11055597
F__inference_model_16_layer_call_and_return_conditional_losses_11055520?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_11054730?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
input_11?????????d
?2?
,__inference_conv2d_28_layer_call_fn_11055833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_28_layer_call_and_return_conditional_losses_11055826?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_48_layer_call_fn_11055967
9__inference_batch_normalization_48_layer_call_fn_11055909
9__inference_batch_normalization_48_layer_call_fn_11055985
9__inference_batch_normalization_48_layer_call_fn_11055891?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055929
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055947
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055871
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055853?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_depthwise_conv2d_10_layer_call_fn_11054828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
Q__inference_depthwise_conv2d_10_layer_call_and_return_conditional_losses_11054818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
9__inference_batch_normalization_49_layer_call_fn_11056043
9__inference_batch_normalization_49_layer_call_fn_11056137
9__inference_batch_normalization_49_layer_call_fn_11056119
9__inference_batch_normalization_49_layer_call_fn_11056061?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056005
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056081
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056099
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056023?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_20_layer_call_fn_11056147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_activation_20_layer_call_and_return_conditional_losses_11056142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_average_pooling2d_33_layer_call_fn_11054918?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
R__inference_average_pooling2d_33_layer_call_and_return_conditional_losses_11054912?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_dropout_26_layer_call_fn_11056176
-__inference_dropout_26_layer_call_fn_11056181?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_26_layer_call_and_return_conditional_losses_11056164
H__inference_dropout_26_layer_call_and_return_conditional_losses_11056159?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_separable_conv2d_10_layer_call_fn_11054944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
Q__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_11054931?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
9__inference_batch_normalization_50_layer_call_fn_11056315
9__inference_batch_normalization_50_layer_call_fn_11056239
9__inference_batch_normalization_50_layer_call_fn_11056333
9__inference_batch_normalization_50_layer_call_fn_11056257?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056295
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056201
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056219
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056277?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_21_layer_call_fn_11056343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_activation_21_layer_call_and_return_conditional_losses_11056338?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_average_pooling2d_34_layer_call_fn_11055034?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
R__inference_average_pooling2d_34_layer_call_and_return_conditional_losses_11055028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_dropout_27_layer_call_fn_11056377
-__inference_dropout_27_layer_call_fn_11056372?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_27_layer_call_and_return_conditional_losses_11056360
H__inference_dropout_27_layer_call_and_return_conditional_losses_11056355?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_flatten_layer_call_fn_11056389?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_layer_call_and_return_conditional_losses_11056383?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_11056409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_11056399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_softmax_layer_call_fn_11056419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_softmax_layer_call_and_return_conditional_losses_11056414?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_11055819input_11"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_11054730? %+,-.?@FGHI^_9?6
/?,
*?'
input_11?????????d
? "1?.
,
softmax!?
softmax??????????
K__inference_activation_20_layer_call_and_return_conditional_losses_11056142h7?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????d
? ?
0__inference_activation_20_layer_call_fn_11056147[7?4
-?*
(?%
inputs?????????d
? " ??????????d?
K__inference_activation_21_layer_call_and_return_conditional_losses_11056338h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
0__inference_activation_21_layer_call_fn_11056343[7?4
-?*
(?%
inputs?????????
? " ???????????
R__inference_average_pooling2d_33_layer_call_and_return_conditional_losses_11054912?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
7__inference_average_pooling2d_33_layer_call_fn_11054918?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
R__inference_average_pooling2d_34_layer_call_and_return_conditional_losses_11055028?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
7__inference_average_pooling2d_34_layer_call_fn_11055034?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055853? M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055871? M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055929r ;?8
1?.
(?%
inputs?????????d
p
? "-?*
#? 
0?????????d
? ?
T__inference_batch_normalization_48_layer_call_and_return_conditional_losses_11055947r ;?8
1?.
(?%
inputs?????????d
p 
? "-?*
#? 
0?????????d
? ?
9__inference_batch_normalization_48_layer_call_fn_11055891? M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
9__inference_batch_normalization_48_layer_call_fn_11055909? M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
9__inference_batch_normalization_48_layer_call_fn_11055967e ;?8
1?.
(?%
inputs?????????d
p
? " ??????????d?
9__inference_batch_normalization_48_layer_call_fn_11055985e ;?8
1?.
(?%
inputs?????????d
p 
? " ??????????d?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056005?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056023?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056081r+,-.;?8
1?.
(?%
inputs?????????d
p
? "-?*
#? 
0?????????d
? ?
T__inference_batch_normalization_49_layer_call_and_return_conditional_losses_11056099r+,-.;?8
1?.
(?%
inputs?????????d
p 
? "-?*
#? 
0?????????d
? ?
9__inference_batch_normalization_49_layer_call_fn_11056043?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
9__inference_batch_normalization_49_layer_call_fn_11056061?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
9__inference_batch_normalization_49_layer_call_fn_11056119e+,-.;?8
1?.
(?%
inputs?????????d
p
? " ??????????d?
9__inference_batch_normalization_49_layer_call_fn_11056137e+,-.;?8
1?.
(?%
inputs?????????d
p 
? " ??????????d?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056201rFGHI;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056219rFGHI;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056277?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_50_layer_call_and_return_conditional_losses_11056295?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
9__inference_batch_normalization_50_layer_call_fn_11056239eFGHI;?8
1?.
(?%
inputs?????????
p
? " ???????????
9__inference_batch_normalization_50_layer_call_fn_11056257eFGHI;?8
1?.
(?%
inputs?????????
p 
? " ???????????
9__inference_batch_normalization_50_layer_call_fn_11056315?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
9__inference_batch_normalization_50_layer_call_fn_11056333?FGHIM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
G__inference_conv2d_28_layer_call_and_return_conditional_losses_11055826k7?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????d
? ?
,__inference_conv2d_28_layer_call_fn_11055833^7?4
-?*
(?%
inputs?????????d
? " ??????????d?
C__inference_dense_layer_call_and_return_conditional_losses_11056399\^_/?,
%?"
 ?
inputs?????????0
? "%?"
?
0?????????
? {
(__inference_dense_layer_call_fn_11056409O^_/?,
%?"
 ?
inputs?????????0
? "???????????
Q__inference_depthwise_conv2d_10_layer_call_and_return_conditional_losses_11054818?%I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
6__inference_depthwise_conv2d_10_layer_call_fn_11054828?%I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
H__inference_dropout_26_layer_call_and_return_conditional_losses_11056159l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
H__inference_dropout_26_layer_call_and_return_conditional_losses_11056164l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
-__inference_dropout_26_layer_call_fn_11056176_;?8
1?.
(?%
inputs?????????
p
? " ???????????
-__inference_dropout_26_layer_call_fn_11056181_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
H__inference_dropout_27_layer_call_and_return_conditional_losses_11056355l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
H__inference_dropout_27_layer_call_and_return_conditional_losses_11056360l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
-__inference_dropout_27_layer_call_fn_11056372_;?8
1?.
(?%
inputs?????????
p
? " ???????????
-__inference_dropout_27_layer_call_fn_11056377_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
E__inference_flatten_layer_call_and_return_conditional_losses_11056383`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????0
? ?
*__inference_flatten_layer_call_fn_11056389S7?4
-?*
(?%
inputs?????????
? "??????????0?
F__inference_model_16_layer_call_and_return_conditional_losses_11055520~ %+,-.?@FGHI^_A?>
7?4
*?'
input_11?????????d
p

 
? "%?"
?
0?????????
? ?
F__inference_model_16_layer_call_and_return_conditional_losses_11055597~ %+,-.?@FGHI^_A?>
7?4
*?'
input_11?????????d
p 

 
? "%?"
?
0?????????
? ?
+__inference_model_16_layer_call_fn_11055695q %+,-.?@FGHI^_A?>
7?4
*?'
input_11?????????d
p

 
? "???????????
+__inference_model_16_layer_call_fn_11055772q %+,-.?@FGHI^_A?>
7?4
*?'
input_11?????????d
p 

 
? "???????????
Q__inference_separable_conv2d_10_layer_call_and_return_conditional_losses_11054931??@I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
6__inference_separable_conv2d_10_layer_call_fn_11054944??@I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
&__inference_signature_wrapper_11055819? %+,-.?@FGHI^_E?B
? 
;?8
6
input_11*?'
input_11?????????d"1?.
,
softmax!?
softmax??????????
E__inference_softmax_layer_call_and_return_conditional_losses_11056414X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_softmax_layer_call_fn_11056419K/?,
%?"
 ?
inputs?????????
? "??????????
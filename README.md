<h2>TensorFlow-FlexUNet-Image-Segmentation-RINGS-Prostate-Glands-Tumor (2025/07/20)</h2>
This is the first experiment of Image Segmentation for Prostate-Glands-Tumor
 based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a 512x512 pixels 
  <a href="https://drive.google.com/file/d/10tgRUWXHYubjv7ZU0g4pwixq1XldfMkr/view?usp=sharing">
RINGS-Prostate-Glands-Tumor-ImageMask-Dataset-V2.zip</a>, which was derived by us from the 
<a href="https://data.mendeley.com/datasets/h8bdwrtnr5/1">
<b>
RINGS algorithm dataset<br>
</b>
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for 512x512 Prostate-Glands-Tumor images</b><br>
As shown below, the inferred masks look similar to the ground truth masks.
The green region represents a prostate gland, and the red a tumor respectively.
 <br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P3_A8_12_16_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P3_A8_12_16_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P3_A8_12_16_1.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P3_C11_13_8_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P3_C11_13_8_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P3_C11_13_8_1.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P5_D4_7_13_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P5_D4_7_13_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P5_D4_7_13_2.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following web site<br>

<a href="https://data.mendeley.com/datasets/h8bdwrtnr5/1">
<b>
RINGS algorithm dataset<br>
</b>
</a>

<br>
Published: 15 April 2021<br>
Version 1<br>
DOI:10.17632/h8bdwrtnr5.1<br>
<br>
<b>Contributors:</b><br>
Massimo Salvi, Martino Bosco, Luca Molinaro, Alessandro Gambella, Mauro Giulio Papotti,<br>
 Udyavara Rajendra Acharya, Filippo Molinari<br>
<br>
<b>Description</b><br>

This repository contains the image dataset and the manual annotations used to develop the RINGS algorithm for automated prostate glands segmentation:<br>
 Salvi M., Bosco M., L. Molinaro, Gambella A., Papotti M., Udyavara Rajendra Acharya, and Molinari F., <br>
"A hybrid deep learning approach for gland segmentation in prostate histopathological images", <br>
Artificial Intelligence in Medicine 2021 (DOI: 10.1016/j.artmed.2021.102076)
<br>
<br>
Background: In digital pathology, the morphology and architecture of prostate glands have been routinely adopted by pathologists to evaluate the presence of cancer tissue. The manual annotations are operator-dependent, error-prone and time-consuming. The automated segmentation of prostate glands can be very challenging too due to large appearance variation and serious degeneration of these histological structures.
Method: A new image segmentation method, called RINGS (Rapid IdentificatioN of Glandural Structures), is presented to segment prostate glands in histopathological images. We designed a novel glands segmentation strategy using a multi-channel algorithm that exploits and fuses both traditional and deep learning techniques. Specifically, the proposed approach employs a hybrid segmentation strategy based on stroma detection to accurately detect and delineate the prostate glands contours.
Results: Automated results are compared with manual annotations and seven state-of-the-art techniques designed for glands segmentation. Being based on stroma segmentation, no performance degradation is observed when segmenting healthy or pathological structures.  Our method is able to delineate the prostate gland of the unknown histopathological image with a dice score of 90.16% and outperforms all the compared state-of-the-art methods.
Conclusions: To the best of our knowledge, the RINGS algorithm is the first fully automated method capable of maintaining a high sensitivity even in the presence of severe glandular degeneration. The proposed method will help to detect the prostate glands accurately and assist the pathologists to make accurate diagnosis and treatment. The developed model can be used to support prostate cancer diagnosis in polyclinics and community care centres. 
<br><br>
<b>Licence</b>: CC BY 4.0<br>
<br>
<br>
<h3>
<a id="2">
2 Prostate-Glands-Tumor ImageMask Dataset
</a>
</h3>
 If you would like to train this Prostate-Glands-Tumor Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/10tgRUWXHYubjv7ZU0g4pwixq1XldfMkr/view?usp=sharing">
RINGS-Prostate-Glands-Tumor-ImageMask-Dataset-V2.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Prostate-Glands-Tumor
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
On the derivation of this dataset, please refer to the following Python scripts.<br>
<li>
<a href="./generator/GlandsTumorImageMaskDatasetGenerator.py">GlandsTumorImageMaskDatasetGenerator.py</a>
</li>
<li>
<a href="./generator/split_master.py">split_master.py</a>
</li>
<br>
<br>
<b>Prostate-Glands-Tumor Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/Prostate-Glands-Tumor_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not large to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Prostate-Glands-Tumor TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00008
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Prostate-Glands-Tumor 3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;rgb color map dict for 1+2 classes
;background:black,  glands:green, tumor: red
rgb_map = {(0,0,0):0, (0,255, 0):1, (255,0,0):2,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 47,48,49)</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 49 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/train_console_output_at_epoch49.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for BUSBRA.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/evaluate_console_output_at_epoch49.png" width="920" height="auto">
<br><br>Prostate-Glands-Tumor

<a href="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Prostate-Glands-Tumor/test was not low and dice_coef_multiclass 
not high as shown below.
<br>
<pre>
categorical_crossentropy,0.524
dice_coef_multiclass,0.7551
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for BUSBRA.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P3_A8_12_16_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P3_A8_12_16_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P3_A8_12_16_1.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P3_C10_7_3_9.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P3_C10_7_3_9.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P3_C10_7_3_9.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P3_C11_13_8_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P3_C11_13_8_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P3_C11_13_8_1.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P5_D4_7_13_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P5_D4_7_13_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P5_D4_7_13_2.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P4_C10_14_8_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P4_C10_14_8_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P4_C10_14_8_2.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/images/P4_C11_16_12_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test/masks/P4_C11_16_12_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Prostate-Glands-Tumor/mini_test_output/P4_C11_16_12_4.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Prostate-Glands-Tumor: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems</b><br>
<b>RINGS algorithm dataset</b><br>
Massimo Salvi, Martino Bosco, Luca Molinaro, Alessandro Gambella, Mauro Giulio Papotti,<br>
 Udyavara Rajendra Acharya, Filippo Molinari<br>
DOI:10.17632/h8bdwrtnr5.1<br>
<a href="https://data.mendeley.com/datasets/h8bdwrtnr5/1">
https://data.mendeley.com/datasets/h8bdwrtnr5/1</a>
<br>

<br>
<b>2.Tensorflow-Image-Segmentation-RINGS-Prostate-Tumor</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-RINGS-Prostate-Tumor">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-RINGS-Prostate-Tumor
</a>
<br>
<br>
<b>3.Tensorflow-Image-Segmentation-RINGS-Prostate-Glands</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-RINGS-Prostate-Glands">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-RINGS-Prostate-Glands
</a>
<br>
<br>
<b>4.Tensorflow-Image-Segmentation-Pre-Augmented-RINGS-Prostate-Tumor</b><br>
<a href="Tensorflow-Image-Segmentation-Pre-Augmented-RINGS-Prostate-Tumor">
Tensorflow-Image-Segmentation-Pre-Augmented-RINGS-Prostate-Tumor</a>
<br>
<br>


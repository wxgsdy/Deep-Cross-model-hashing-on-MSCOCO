# DCMH_on_MSCOCO
## Description: Design and Implementation of Deep Hashing Algorithm for Cross-modal Retrieval
• Proposed a deep hash algorithm for cross-modal retrieval of text and images through integrating feature learning and 
hash-code learning into the same framework.  
• Realized the algorithms framework with two separate deep neural networks (one for image modality and the other for                 
text modality) on MATLAB with MatConvNet toolbox.  
• Tested the algorithm on Microsoft COCO dataset, represented the images and annotations with hash code, improved 
cross modal data retrieval precision.  

## File description
1. MAPTool floder: tool functions like mean average precision to evaluate the model.
2. matconvnet-beta23: Released version of matlab CNN toolbox from VlFeat.
3. Glove: word embedding tool like word2vector.
4. "DCMH.m": whole process including data preparation, training and testing.
5. "net_structure_image.m","net_structure_text.m": image and text CNN definition.
6. "update_image.m","update_text.m": update net parameters with tricks like momentum and weight decay.
7. "test.m" : test trained model on testing set.
8. other files are matlab function files called by these main file above.  

### Usage
1. Run Glove to generate word vector for frequent words in MSCOCO Annotations dataset.
2. Run train.m wihch includes all the process like data prepraring, train the model and test.

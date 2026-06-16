# WOUNDCARE VQA DATASET

This data includes consumer health queries with both text and images, as well as responses by US medical doctors.


## Data Description

Each query instance includes:
1. encounter_id - a unique identifier (e.g. 'ENC0001')
2. query_title_{LANG} - the query title
3. query_content_{LANG} - the query content
4. image_ids - a list of one or more image id's (e.g. [ 'IMG_ENC0001_0001.jpg', 'IMG_ENC0001_0002.jpg' ])
5. responses - a list of one more more response objects with fields: author_id, content_{LANG}

This dataset includes two languages English (en), and Chinese (zh). Translations to and from English and Chinese are conducted by bilingual translators with backgrounds in nursing or medical translation.


## Additional Metadata

Additional metadata related to the source information as well as additional medically related metadata include:
1. src - source of dataset (e.g. baiduzhidao, tieba)
2. post_id - post identifier, naming convention is {src}_{original post ID}
3. post_url - URL of the original query
4. anatomic_locations - list of anatomic locations
5. wound_type - wound type category
6. wound_thickness - wound thickness category
7. tissue_color - tissue color category
8. drainage_amount - drainage amount category
9. drainage_type - drainage type category
10. infection - infection category

Original src, post_id, post_url data is not included in the general challenge datasets.


## Data Statistics

| split | instances | responses | images
|---|---|---|---|
| train | 279 | 279 | 449 |
| valid | 105 | 210 | 147 |
| test | 93 | 279 | 152 |

The train split has one one response, valid is double annotated, test is tripe-annotated, by seperate individual medical doctors respectively.

Using the sacrebleu tokenizers -- tokenizer_13a for English and tokenizer_zh for Chinese, below are general information related to the corpus text.

### Average # Words - ENGLISH (en)
| split | query | response |
|---|---|---|
| train | 46 | 29 |
| valid | 44 | 41 |
| test | 52 | 47 |

### Average # Characters - CHINESE (zh)
| split | query | response |
|---|---|---|
| train | 52 | 43 |
| valid | 50 | 60 |
| test | 60 | 68 |


## Metadata Agreement

For a small subset (37 instances), all three medical doctor annotators labeled the metadata items. The following metrics give their agreement.

### Strict Agreement (Accuracy)

#### Annotator 1&2:
| category | agreement |
|---|---|
| anatomic_locations | 0.4864864864864865 |
| wound_type | 0.8918918918918919 |
| wound_thickness | 0.40540540540540543 |
| tissue_color | 0.6216216216216216 |
| drainage_amount | 0.5135135135135135 | 
| drainage_type | 0.6486486486486487 |
| infection | 0.7837837837837838 |

#### Annotator 1&3:
| category | agreement |
|---|---|
| anatomic_locations | 0.4864864864864865 |
| wound_type | 0.8648648648648649 |
| wound_thickness | 0.32432432432432434 |
| tissue_color | 0.5675675675675675 |
| drainage_amount | 0.5405405405405406 |
| drainage_type | 0.6756756756756757 |
| infection | 0.7027027027027027 |

#### Annotator 2&3:
| category | agreement |
|---|---|
| anatomic_locations | 0.8108108108108109 |
| wound_type | 0.918918918918919 |
| wound_thickness | 0.6486486486486487 |
| tissue_color | 0.7567567567567568 |
| drainage_amount | 0.7297297297297297 |
| drainage_type | 0.8378378378378378 |
| infection | 0.7297297297297297 |


### Relaxed Agreement (At least 2 votes)

| category | agreement |
|---|---|
| anatomic_locations | 0.8108108108108109 |
| wound_type | 1.0 |
| wound_thickness | 0.8918918918918919 |
| tissue_color | 0.972972972972973 |
| drainage_amount | 0.8648648648648649 |
| drainage_type | 0.918918918918919 |
| infection | 0.972972972972973 |

//For anatomic location, if each annotator's entry overlaps with the other two, only then is a match counted.


## File Description

### Challenge Related Files

1. train.json - This file includes the query/responses for the train set.
2. valid.json - This file includes the query/responses for the valid set.
3. valid_inputonly.json - This file includes the query inputs for the valid set.
4. test.json - This file includes the query/responses for the test set.
5. test_inputonly.json - This file includes the query inputs for the test set.
6. images_train.zip - This file contains the train image files.
7. images_valid.zip - This file contains the valid image files.
8. images_test.zip - This file contains the test image files.

### Full Dataset Files

1. woundcarevqa.json - This file includes the full corpus (including all splits). Both original (post_url) and anonymized ids (encounter_ids) are included. Single-annotated medical metadata from annotator1 is also included.

2. data_dictionary.txt - Text sheet of metadata descriptions for medically annotated medical data originally given as labeling guidelines.

3. woundcarevqa_metadata.xlsx - Excel sheet with metadata triple annotated.

4. images_final.zip - This contains all the image files.


This dataset is only for scientific research purposes. If you feel this dataset infringes upon your copyright, please notify yimwenwai AT microsoft.com
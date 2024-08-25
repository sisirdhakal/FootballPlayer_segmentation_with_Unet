# Football Player Segmentation with UNet

This project demonstrates semantic segmentation of football players using a custom UNet architecture implemented from scratch. The model was trained, validated, and tested on a processed dataset obtained from Kaggle.

## Dataset

The dataset used in this project is the [Football Player Segmentation Dataset](https://www.kaggle.com/datasets/ihelon/football-player-segmentation) available on Kaggle. The dataset contains images and masks that represent the segmentation of football players.

## Dataset Preparation

The dataset was processed to create three folders: `train`, `test`, and `val`, following an 80-10-10 split ratio:

- `train`: 80% of the dataset
- `val`: 10% of the dataset
- `test`: 10% of the dataset

Each folder contains images and corresponding masks for training, validation, and testing purposes.

## Project Overview

This project was developed by implementing all necessary functions and the UNet architecture from scratch. The UNet model was trained to perform semantic segmentation, specifically identifying and segmenting football players from the background in the images.

### Key Features

- **Custom UNet Architecture**: The UNet model was implemented from scratch, with custom layers and modules.
- **Data Augmentation**: Various augmentation techniques were applied to improve the model's generalization.
- **Training and Validation**: The model was trained using the training set, and its performance was validated using the validation set.
- **Testing and Evaluation**: After training, the model was evaluated on the test set, and results were visualized.

## Results

The model's performance was evaluated using metrics such as loss and accuracy. The results were visualized, showing the segmentation of football players in test images.

## Conclusion

This project successfully demonstrates the ability to perform semantic segmentation of football players using a custom-built UNet model. The implementation from scratch allows for a deep understanding of the architecture and the process of semantic segmentation.

## References

- [Football Player Segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/ihelon/football-player-segmentation)
- [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## License

This project is open-source and available under the MIT License.

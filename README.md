# Project for ID2223 Course @ KTH

**Keywords:** scalable machine learning, data engineering, classification, model ensemble
<p align="center">
    <img src="https://raw.githubusercontent.com/GoodOnions/ID2223-Lab1/main/imgs/goodonions_cover.png" alt="GoodOnions Official Repository"/>
</p>

## Team

**Name:** GoodOnions\
**Components:** [Daniele Cipollone](https://github.com/dancip00), [Federico Bono](https://github.com/FredBonux)

## Project report
Navigating the salary landscape as a recent graduate can pose challenges, especially when determining an appropriate salary range for a specific position. Our project focuses on utilizing machine learning (ML) to develop a tool that aids new data scientists in understanding and establishing reasonable salary expectations for their roles.

We have successfully crafted a publicly accessible inference platform capable of providing a reasonable salary range with minimal input, guiding individuals during their job search.

Following the completion of an exploratory data analysis (EDA), we identified the most influential features in the dataset. Subsequently, we trained and evaluated various regression and classification models (for salary ranges) on the initial dataset, culminating in the selection of the most performant model. This chosen model is seamlessly integrated into an inference interface, allowing users to obtain personalized salary predictions.

Simultaneously, we've implemented Hopsworks for efficient data and model storage, utilized GitHub Actions for the annual execution of our data fetching processes and training pipeline. For the application interface, we've harnessed the power of Hugging Face integrated with Gradio.

To ensure the continuous relevance of our data, we've established a robust yearly fetching infrastructure. This comprehensive approach covers the entire spectrum from data analysis to model deployment, creating a streamlined and effective system for personalized salary predictions.

## Index
1. [Exploratory data analysis (EDA)](./salary/salary-eda.ipynb)
2. [Model training](./salary/salary-training-pipeline.ipynb)
3. [Data fetching pipeline (yearly)](./salary/salary-yearly.py)
4. [Hugging face application](https://huggingface.co/spaces/GoodOnions/ID2223-Project)

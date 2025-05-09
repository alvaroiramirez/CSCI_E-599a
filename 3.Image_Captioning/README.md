# Disaster Image Captioning with Multi-Modal Models

This repository contains the methodology and resources for fine-tuning multi-modal models to generate accurate captions for disaster imagery. This project aims to enhance disaster assessment processes by automating image captioning while maintaining high accuracy.

## About This Project

Disaster response teams rely on rapid assessment of imagery to make critical decisions. By fine-tuning state-of-the-art vision-language models, we can dramatically reduce the time required for image analysis while ensuring consistent assessment quality.

## Supported Models

We've documented approaches for fine-tuning three leading multi-modal models:

### OpenAI GPT-4o
- API-based fine-tuning approach
- Excellent visual understanding capabilities
- Deployed via OpenAI's infrastructure
- Best for scenarios where control over infrastructure isn't required

### Meta's Llama 3.2
- Open-weight model allowing full customization
- Self-hosted deployment options
- Strong performance with proper fine-tuning
- Ideal for scenarios requiring infrastructure control or offline deployment

### Google's PaLI-GEMMA
- Purpose-built for image understanding tasks
- Efficient architecture for deployment
- Balance of performance and resource requirements
- Good option for resource-constrained environments

## Data Preparation

Effective fine-tuning requires quality data with consistent formatting:

- **Images**: Cropped to focus on damage areas (minimum 100×100 pixels)
- **Captions**: Structured format with damage classification followed by description
- **Example**: "Major damage, residential building with collapsed roof and structural failure"

We recommend preparing at least 100 high-quality image-caption pairs for training and 25 pairs each for validation and testing.

## Fine-Tuning Process

The general fine-tuning workflow follows these steps:

1. **Data Preparation**: Organize and format image-caption pairs
2. **Model Selection**: Choose the appropriate model based on your requirements
3. **Fine-Tuning Configuration**: Set hyperparameters for your specific case
4. **Training Execution**: Perform the fine-tuning process
5. **Evaluation**: Assess performance using standard metrics
6. **Deployment**: Implement the model in your workflow

Each model has specific fine-tuning requirements documented in the respective directories.

## Evaluation Framework

We recommend evaluating fine-tuned models using:

- **Automated Metrics**: Cosine Similarity from BERT Embeddigns
- **Human Evaluation**:  Assessment of caption quality
- **Performance Testing**: Processing time and resource usage

## Implementation Considerations

When implementing a fine-tuned model, consider:

- **Batch Processing**: For efficient handling of large image sets
- **Integration**: Connection to existing assessment workflows
- **Confidence Thresholds**: Identifying cases requiring human review
- **Feedback Loop**: Mechanism for continuous improvement

## Resources

- `Final Report/`: Detailed methodology documentation
- `Dataset/`: Construct the dataset frm current or newe data
- `Models/`: Notebooks for fine-tuning
- `Evaluation/`: Notebook for evalauting with images to inference

## Contributors

FEMA Project Team

## License

This project is licensed under the MIT License - see the LICENSE file for details.



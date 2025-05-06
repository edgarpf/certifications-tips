# AWS Certified AI Practitioner
* Artificial Intelligence is a field of computer science dedicated to solving problems that we commonly associate with human intelligence.
* Artificial Intelligence >  Machine Learning > Deep Learning > Generative AI.
* Generative AI is used to generate new data that is similar to the data it was trained on. To generate data, we must rely on a Foundation Model. Foundation Models are trained on a wide variety of input data. The models may cost tens of millions of dollars to train.
* Large Language Models (LLM) is the type of AI designed to generate coherent human-like text (Chat GPT).
* Non-deterministic: the generated text may be different for every user that uses the same prompt.
* Amazon Bedrock
  * Build Generative AI (Gen-AI) applications on AWS.
  * Fully-managed service, no servers for you to manage.
  * Keep control of your data used to train the model.
  * Access to a wide range of Foundation Models (FM)
* Amazon Titan
  * High-performing Foundation Models from AWS.
  * Can be customized with your own data.
* Automatic Evaluation vs Human Evaluation.
* Business Metrics to Evaluate a Model On: User Satisfaction, Average Revenue Per User (ARPU), Cross-Domain Performance, Conversion Rate and Efficiency.
* Amazon Bedrock – Guardrails
  * Control the interaction between users and Foundation Models (FMs).
  * Filter undesirable and harmful content.
  * Remove Personally Identifiable Information (PII).
  * Reduce hallucinations
* Prompt Engineering = developing, designing, and optimizing prompts to enhance the output of FMs for your needs.
* Negative Prompting is a technique where you explicitly instruct the model on what not to include or do in its response.
* Retrieval-Augmented Generation (RAG) - Combine the model’s capability with external data sources to generate a more informed and contextually rich response.
* Amazon Q Business- Fully managed Gen-AI assistant for your employees. Based on your company’s knowledge and data.
* Amazon Q Apps - Create Gen AI-powered apps without coding by using natural language.
* Amazon Q Developer - Answer questions about the AWS documentation and AWS service selection. Answer questions about resources in your AWS account.
* Deep Learning - Uses neurons and synapses (like our brain) to train a model.
* Supervised Learning
  * Learn a mapping function that can predict the output for new unseen input data.
  * Needs labeled data: very powerful, but difficult to perform on millions of datapoints.
  * Regression - Used to predict a numeric value based on input data.
  * Classification - Used to predict the categorical label of input data.
* Feature Engineering - The process of using domain knowledge to select and transform raw data into meaningful features.
* Unsupervised Learning - The goal is to discover inherent patterns, structures, or relationships within the input data.
* Reinforcement Learning - A type of Machine Learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards.
* • Inferencing is when a model is making prediction on new data.
* AWS AI Services are pre-trained ML services for your use case.
* Amazon Comprehend - Uses machine learning to find insights and relationships in text.
* Amazon Translate.
* Amazon Transcribe.
* Amazon Polly - Turn text into lifelike speech using deep learning.
* Amazon Rekognition - Find objects, people, text, scenes in images and videos using ML.
* Amazon Forecast - Fully managed service that uses ML to deliver highly accurate forecasts.
* Amazon Lex & Connect - same technology that powers Alexa. Receive calls, create contact flows, cloud-based virtual contact center.
* Amazon Personalize - Fully managed ML-service to build apps with real-time personalized recommendations.
* Amazon Textract - Automatically extracts text, handwriting, and data from any scanned documents using AI and ML.
* Amazon Kendra - Fully managed document search service powered by Machine Learning.
* Amazon Mechanical Turk - Crowdsourcing marketplace to perform simple human tasks.
* Amazon Augmented AI (A2I) - Human oversight of Machine Learning predictions in production.
* AWS DeepRacer.
* Amazon Transcribe Medical - Automatically convert medical-related speech to text.
* Amazon Comprehend Medical - Amazon Comprehend Medical detects and returns useful information in unstructured clinical text.
* Amazon SageMaker - Fully managed service for developers / data scientists to build ML models.
* SageMaker Clarify - Evaluate Foundation Models. Ability to detect and explain biases in your datasets and models
* SageMaker Canvas - Build ML models using a visual interface (no coding required).
* SageMaker Automatic Model Tuning: tune hyperparameters
* SageMaker Deployment & Inference: real-time, serverless, batch, async
* SageMaker Studio: unified interface for SageMaker
* SageMaker Data Wrangler: explore and prepare datasets, create features
* SageMaker Feature Store: store features metadata in a central place
* SageMaker Ground Truth: RLHF, humans for model grading and data labeling
* SageMaker Model Cards: ML model documentation 
* SageMaker Model Dashboard: view all your models in one place 
* SageMaker Model Monitor: monitoring and alerts for your model
* SageMaker Role Manager: access control
* SageMaker JumpStart: ML model hub & pre-built ML solutions
* MLFlow on SageMaker: use MLFlow tracking servers on AWS
* AWS Macie - Amazon Macie is a fully managed data security and data privacy service that uses machine learning and pattern matching to discover and protect your sensitive data in AWS.
* Amazon Inspector - Automated Security Assessments. Only for EC2 instances, Container Images & Lambda functions
* AWS Artifact -  Portal that provides customers with on-demand access to AWS compliance documentation and AWS agreements.
* Neural networks are utilized as deep learning models that simulate the human brain’s pattern recognition capabilities, learning from historical financial data to anticipate future stock market trends.
* A multimodal model is an advanced machine learning architecture that can process and integrate information from multiple types of data or modalities. In the context of artificial intelligence and machine learning, modalities refer to different forms of data inputs, such as text, images, audio, and video. Multimodal models are designed to handle these diverse data types simultaneously, enabling a more comprehensive understanding and interpretation of complex information. For instance, in healthcare, combining text data from patient reports with image data from medical scans can provide a richer and more accurate diagnostic capability.
* Amazon SageMaker Model Monitor is a capability of Amazon SageMaker that monitors machine learning models in production for data drift, concept drift, and other issues that may impact model quality. It continuously monitors the data inputs and model predictions to detect deviations from the model’s expected behavior. Model Monitor can automatically alert users when it detects issues, allowing them to take corrective actions.
* Intelligent Document Processing (IDP) automates data processing using OCR, computer vision, NLP, and machine learning. It extracts, categorizes, and generates insights from unstructured data. IDP enhances customer satisfaction and operational efficiency through generative AI-powered automation. Its ready-to-use APIs efficiently process unstructured data at scale, extract critical information, and generate insightful summaries and reports.
* The National Institute of Standards and Technology (NIST) develops comprehensive recommendations and standards for US federal information systems. These principles assure the confidentiality, integrity, and availability of information, making them necessary for federal regulatory compliance.
* Amazon SageMaker Clarify is an essential tool for machine learning specialists who are concerned about bias and transparency in their models. It helps detect bias at multiple stages, including during data preparation, after model training, and even during inference. By evaluating datasets and models for potential biases, SageMaker Clarify ensures that machine learning models are fair and do not unintentionally disadvantage any particular group. This capability is crucial for building ethical and reliable models that can be trusted by stakeholders and users alike.
* Role-based access controls restrict data access to only authorized personnel, preventing unauthorized individuals or systems from accessing or modifying the sensitive training data.
* Secure data engineering practices are essential for ensuring the safety and reliability of AI and generative AI systems. The following are some best practices to consider.
  * Assessing Data Quality: Data quality assessment is the process of evaluating the fitness, accuracy, completeness, consistency, and reliability of data used for training and deploying AI/ML models. It involves identifying and mitigating potential issues such as biases, errors, inconsistencies, and missing values in the data.
  * Implementing Privacy-Enhancing Technologies: Privacy-enhancing technologies are a set of techniques and tools designed to protect personal data and individual privacy while allowing data to be used for legitimate purposes, such as training AI/ML models. Examples of PETs include data anonymization, pseudonymization, encryption, tokenization, differential privacy, and secure multi-party computation.
  * Data Access Control: Data access control refers to the mechanisms and policies that govern who can access, modify, or delete data used in AI/ML systems. It involves implementing role-based access controls, authentication, and authorization measures to ensure that only authorized individuals or services can access sensitive data.
  * Data Integrity: Data integrity refers to the accuracy, completeness, and consistency of data throughout its lifecycle, from ingestion to processing, storage, and analysis. It involves implementing measures to protect data from unauthorized modifications, accidental or malicious corruption, and ensuring that data remains trustworthy and reliable for AI/ML applications.
* Managing customer satisfaction (CSAT) data is crucial for continuously improving service quality. AWS provides tools and resources for managing CSAT data in APN Partner Central, allowing organizations to gain insights and take action based on customer feedback. This ensures that the service meets customer expectations and enhances overall satisfaction.
* AWS AI Service Cards are responsible AI documentation that provides customers with a single source of information on the intended use cases and constraints, responsible AI design choices, and deployment and performance optimization best practices for AWS AI services.
* To use SageMaker: The first step is to ensure that the dataset of customer interactions is available in Amazon S3, as this is where Amazon SageMaker will access the data for training. Once the dataset is uploaded, the next step is to create a training job in Amazon SageMaker. This involves specifying the algorithm, the compute resources, and other configurations needed for training the chatbot model. Finally, you need to configure the training job to use the dataset stored in Amazon S3. 
* Responsible AI refers to the practice of developing and deploying AI systems in a way that promotes ethical principles, such as fairness, transparency, privacy, and accountability. It aims to mitigate potential risks and negative impacts associated with AI systems.
  * Transparency refers to the ability to understand and inspect the inner workings, decision-making processes, and outputs of an AI system. It involves making the AI system’s behavior, decisions, and underlying logic visible and comprehensible to relevant stakeholders, such as developers, regulators, and end-users.
  * Explainability: Providing clear explanations and interpretability of how AI systems make decisions, enabling accountability and trust.
  * Veracity and Robustness: Ensuring AI systems operate reliably and consistently and are resilient to potential failures or adversarial attacks.
  * Fairness: Ensuring AI models are unbiased and do not discriminate against individuals or groups based on protected characteristics.
  * Privacy and Security: Safeguarding the privacy and security of the data used by AI systems and handling personal information responsibly.
  * Governance: Establishing clear governance frameworks, policies, and processes for the responsible development and deployment of AI systems.
  * Safety: Identifying and mitigating potential risks and unintended consequences associated with AI systems.
  * Controllability Maintaining appropriate human control and oversight over AI systems, particularly in high-stakes decision-making scenarios.
* Regression is a supervised learning technique used for predicting continuous values. It involves determining the relationship between a dependent variable and one or more independent variables
* Linear regression refers to supervised learning models that use one or more inputs to predict a value on a continuous scale. It is used to predict housing prices.
* Large language models (LLMs) are extensive deep learning models pre-trained on massive datasets. They utilize a transformer architecture, which includes neural networks composed of an encoder and a decoder with self-attention mechanisms. These components work together to derive meaning from text sequences and comprehend the relationships between words and phrases.
* Stable Diffusion is a generative AI model that creates distinctive photorealistic images based on text and image prompts.
* Multimodal models are AI systems that can process and generate content across multiple modalities, such as text, images, and audio. These models are designed to understand and integrate information from diverse data types, enabling more comprehensive and contextually rich outputs.
* Researchers introduced the term “foundation model” to describe machine learning models that are trained on a diverse range of generalized and unlabeled data. These models are capable of performing a wide array of tasks, including language comprehension, text and image generation, and natural language conversation.
* Model Evaluation on Amazon Bedrock enables you to assess, compare, and choose the most suitable foundational models for your specific needs. Amazon Bedrock provides the option of automatic evaluation and human evaluation.
* Challenges of Generative IA:
  * Toxicity
  * Hallucinations
  * Intellectual Property
  * Plagiarism and cheating
  * Disruption of nature work
* Amazon SageMaker Model Parallelism is a feature designed to help train large deep-learning models that cannot fit into the memory of a single GPU. This feature automatically partitions the model across multiple GPUs, efficiently training very large models.
* Prompt engineering is the process of designing and crafting prompts (input text) to guide language models to generate desired outputs. Some of the Prompt Engineering Techniques are:
  * Few-shot prompting is a technique that involves providing a language model with contextual examples to guide its understanding and expected output for a specific task. This is technique is particularly helpful for models with limited training data or when adapting the model to new domains or tasks.
  * Chain-of-thought (CoT) prompting is a technique that divides intricate reasoning tasks into smaller intermediary steps. This technique can help the model reason through the task in a structured manner and improve its ability to handle complex tasks.
  * Zero-shot prompting is a technique where a user presents a task to a generative model without providing any examples or explicit training for that specific task. Instead, it relies solely on the general knowledge acquired during pre-training to perform the task based on the prompt.
* Sampling Bias is a common issue in machine learning that occurs when the training data used to develop a model does not represent the entire population. This leads to biased outcomes because the model learns patterns that reflect the imbalances in the data rather than accurately capturing the relationships within the whole population.
* Reporting bias occurs when certain data points or outcomes are selectively reported or emphasized while others are omitted.
* Automation bias refers to the tendency of people to over-rely on automated systems and trust their decisions without questioning their accuracy or fairness.
* Implicit Bias is incorrect because it refers to unconscious attitudes or stereotypes that affect human decision-making.
* Capabilities of Generative IA:
  * Responsiveness - Generates content in real time.
  * Adaptability - Adap various questions and domains.
  * Scalability - Generate large amounts of data quickly.
  * Personalization - Create personalized content.
  * Data efficiency - Learn from small amount of data.
  * Creativity and exploration - Generate new ideas.
  * Simplicity - Simplify complex tasks.
* Embeddings are a way of representing words or phrases as dense vectors of real numbers, where semantically similar words or phrases are represented by vectors that are close together in the vector space.
* ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics specifically designed to evaluate the quality of text summaries by comparing them to human reference summaries.
* Bidirectional Encoder Representations from Transformers (BERT), a bidirectional model, examines the context of an entire sequence before making predictions. It was trained on a plain text corpus and Wikipedia, utilizing 3.3 billion tokens (words) and 340 million parameters. BERT is capable of answering questions, predicting sentences, and translating texts.
* Vector search is a method used in machine learning to find similar data points to a given data point by comparing their vector representations using distance or similarity metrics.
* Another limitation of generative AI is the knowledge cutoff. Generative AI models are trained on datasets available up to a certain point in time, and they do not have access to real-time data or updates
* Forecasting algorithms are crucial in predictive analytics, especially for predicting future values based on historical patterns in time series data. AWS offers strong forecasting capabilities through Amazon SageMaker Canvas.
* The model monitoring system must capture data, compare that data to the training set, define rules to detect issues and send alerts. This process repeats on a defined schedule when initiated by an event or when initiated by human intervention. The issues detected in the monitoring phase include data quality, model quality, bias drift, and feature attribution drift.
* Amazon SageMaker offers various inference options tailored to different workload needs:
  * Real-time Inference: Used for low-latency, predictable traffic patterns requiring consistent availability. It is ideal when the service needs to be always available.
  * Serverless Inference: Suitable for workloads with spiky traffic patterns that can tolerate latency variations. It automatically scales, and you only pay during inference requests, making it cost-effective for unpredictable usage.
  * Batch Inference: Best for offline processes that require inference on large datasets. You pay only for the duration of the job, making it ideal when continuous availability is not needed.
* Amazon SageMaker Feature Store is a fully managed service that allows organizations to centrally store, share, and manage machine learning features across multiple projects.
* Amazon SageMaker Data Wrangler is a SageMaker feature designed to simplify the data preparation process for machine learning workflows.
* The DetectModerationLabels API is specifically designed for content moderation. It analyzes images to identify unsafe or inappropriate content, such as nudity, violence, or suggestive material. This API’s primary purpose is to detect moderation labels, making it suitable for filtering out unsafe content.
* Amazon SageMaker Clarify is designed to help machine learning practitioners address fairness, bias, and explainability in their models.
* Amazon SageMaker Model Cards provide a structured framework for documenting essential details about your machine-learning models at every stage of their development.
* Amazon SageMaker Ground Truth is a service provided by AWS that enables data labeling for machine learning projects. It allows you to create labeling jobs and manage a workforce of human labelers to annotate your data.
* Overfitting is a machine learning phenomenon in which a model performs well on training data but struggles to predict new, previously unseen data accurately. This combination of low bias and high variance suggests that the model is too sensitive to the unique properties of the training data.
* The foundational capability required to ensure the effective and responsible use of generative AI is the Governance Perspective.
* Instruction-based fine-tuning is a process where a pre-trained foundation model is further trained with specific instructions to perform particular tasks.
* Domain adaptation fine-tuning allows you to leverage pre-trained foundation models and adapt them to specific tasks using limited domain-specific data.
* Amazon SageMaker JumpStart is a machine learning (ML) hub designed to accelerate your ML journey. With SageMaker JumpStart, you can quickly evaluate, compare, and select pre-trained machine learning models based on predefined quality and reliability metrics.
* Embeddings are numerical representations of real-world objects that machine learning (ML) and artificial intelligence (AI) systems use to understand complex knowledge domains like humans.
* A confusion matrix is a tool for visualizing the performance of a multiclass model. It has entries for all possible combinations of correct and incorrect predictions, and shows how often each one was made by our model.
* MAPE (Mean Absolute Percentage Error) calculates the average of the absolute differences between actual and projected values, divides it by actual values, and returns a percentage.
* MAE (Mean Absolute Error) is the average difference between expected and actual values for all observations.
* Underfitting is a type of error that occurs when the model cannot determine a meaningful relationship between the input and output data. You get underfit models if they have not trained for the appropriate length of time on a large number of data points. The model performs poorly on the training and validation datasets, showing high bias.
* Amazon SageMaker Ground Truth provides a wide range of human-in-the-loop capabilities, enabling you to leverage human feedback throughout the machine learning process to enhance model accuracy and relevance.
* One of the key steps in the RLHF approach is creating a reward model. The reward model is trained on human feedback to learn to predict the quality or appropriateness of the language model’s outputs. Another key step in the RLHF approach is fine-tuning the language model using the reward model and reinforcement learning techniques.
* BERTScore is a tool that compares how similar generated text is to a reference by understanding the context of words leveraging BERT (Bidirectional Encoder Representations from Transformers) embeddings.
* Amazon SageMaker Model Registry is designed specifically to manage machine learning models.
* Amazon Rekognition offers a Content Moderation feature that can detect explicit or suggestive content in images and videos. It uses machine learning models to identify potentially explicit content based on various criteria, such as nudity, violence, or offensive language.
* Length of the input/output data sequence can directly influence the latency of a machine learning model’s inference.
* Amazon RDS for PostgreSQL supports the pgvector extension, which is designed for storing embeddings from machine learning models and performing efficient similarity searches.
* Binary classification is a supervised machine learning model specifically designed to distinguish between two distinct categories or classes. This model is widely used in various applications, such as sentiment analysis, fraud detection, and medical diagnosis, where the objective is to classify data points into one of two predefined categories.
* ROUGE-N is a widely recognized metric for evaluating text summarization models, including those generated by foundation models.
* Data augmentation is a technique for expanding a dataset by generating new samples from existing data through various transformations
* Accuracy is a commonly used metric to evaluate the performance of classification models in machine learning. It measures the proportion of correctly classified instances from the total instances in the dataset, providing a general sense of how well the model performs across all classes.
* Decision trees are a popular machine learning algorithm known for their interpretability and simplicity. They operate by recursively splitting the data based on features like traffic conditions, weather data, and route information. This creates a tree-like structure where each node represents a decision based on a specific feature. This hierarchical structure allows decision trees to clearly illustrate how different factors influence the outcome, making it easy to understand and interpret the predictions.
* Amazon SageMaker Batch Transform is an efficient and cost-effective way to make predictions across an entire dataset. It is suitable for use cases where immediate access to predictions is not required, and the data can be processed in batches.
* A context window is the amount of text an AI model can handle and respond to at once; in most LLMs, this text is measured in tokens.
* Real-time inference is best suited for inference workloads with real-time, interactive, and low-latency requirements.
* Generative Adversarial Networks (GANs) are a type of machine learning model designed to generate new data by learning from an existing dataset.
* The number of tokens processed during inference directly impacts the cost because pricing in services like Amazon Bedrock is often based on the amount of data (tokens) being processed.
* Hyperparameters are configuration settings that are set before the training process begins and control the behavior of the machine learning algorithm. These settings are not learned from the data but are tuned by the developer or data scientist to optimize the model’s performance.
* An epoch is a single pass through the entire training dataset. During each epoch, the model sees the entire dataset once and updates its internal parameters (weights and biases) based on the errors it makes on the training data.
* Average response time is simply the time it takes for a model to process an input and give an output.
* Traditional Machine Learning (ML) models use structured data and predetermined algorithms to predict or classify. These models often require human intervention for feature engineering and are ideal for tasks such as regression, classification, and grouping. Traditional machine learning models include linear regression, decision trees, and logistic regression.
* Amazon EC2 Trn1 Instances are specifically designed for high-performance machine learning (ML) model training and utilize AWS Trainium chips.
* The Generative Pre-trained Transformer is a language model that understands and generates natural language.
* A multi-modal embedding model is a foundation model that integrates and processes different types of data, such as text, images, and other forms of structured data. This model is highly useful in scenarios that require analyzing and understanding multiple data modalities simultaneously.
* Increasing the number of epochs allows the model to train for more iterations, which provides more opportunities for the model to learn from the data and improve its accuracy.
* Partial Dependence Plots (PDPs) are a powerful tool for explaining machine learning models by showing the relationship between features and the model’s predictions.
* Exploratory Data Analysis (EDA) is the process of analyzing and understanding the characteristics of the data before building an ML model. It involves tasks such as visualizing data distributions, calculating summary statistics, identifying missing values, and detecting outliers. EDA aims to gain insights into the data and identify potential issues or patterns that may impact the model’s performance.
* The object detection algorithm can identify and locate all instances of objects in an image from a known collection of object categories.
* Amazon Rekognition provides content moderation APIs that can analyze images and detect inappropriate content, such as nudity, violence, or offensive symbols.
* Amazon Bedrock offers a seamless experience for deploying and utilizing pre-trained foundation models. By activating invocation logging within Amazon Bedrock, the service records both the input and output data for every invocation, providing detailed logs that can be monitored for accuracy, performance, or troubleshooting.

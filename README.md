This was a submission that utilizes a machine learning pipeline for gas classification for the Inter-Uni 26 Datathon's Stream 3.

The train.csv was splitted into 2 into train_version_1.csv and train_version_2.csv for experimentation with the data. Splitting the training CSV into two versions did not directly improve accuracy.

Validation Strategy
Due to the competition's time constraints and hidden test labels, traditional validation methods were supplemented with public leaderboard evaluation. Multiple preprocessing techniques, feature representations, PCA configurations, and model architectures were tested and compared. Public leaderboard performance was used as a practical benchmark for model selection, while maintaining a consistent experimental process and avoiding manual modification of predictions.

Reproduce Files:

Dependencies:
- pandas
- numpy
- scipy
- scikit-learn

To reproduce the results, change the file directory after downloading them via 

git https://github.com/Polypholia/Interuni-Datathon-Perms-Combs/tree/main. 
  
Then run via python3 mL_FinalSubmission.py.



Copilot and ChatGPT 5 were used to help debug and find formulate strategies and techniques.

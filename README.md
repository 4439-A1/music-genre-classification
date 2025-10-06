# Music Genre Classification
 This project classifies 30-second clips of music into one of 10 genres. The dataset consists of 10 genres with equal number of audio files. Each audio file has a length of 30 seconds. The label for each audio file is present in a CSV file called train.csv. The dataset is split into 80% training data, 10% testing data that is publicly available, and 10% private testing data. On Kaggle, where I submitted my results, I can only see my performance on the public testing data. The final classification accuracy, however, depends on both the public and private testing data.

 See Music_Genre_Classification_Report.pdf for more details of the model's performance. The report explains all the methods/models that I tried for feature extractions and classification. In the folder Testing Code, there are code files which I used throughout the way but ended up discarding because they didn't give the highest accuracy. These code files are referenced in the report. The folder Final Submission Code contains the code that I used for the Kaggle submissions.

 **Note:** On Safari, Music_Genre_Classification_Report.pdf needs to be downloaded before it can be viewed.

# 音楽ジャンルの識別を行うプログラム
このプロジェクトでは30秒の長さの音声ファイルを10の音楽ジャンルのうちの一つに振り分けるプログラムを開発しました。データセットには10のジャンルに当たる音楽のファイルが均等に含まれていて、各ファイルが30秒のwavファイルとなっています。train.csvには各ファイルのジャンルが記載されています。データせっとの80％をトレーニングデータとして扱って、10％はテスティングデータとして扱って使い分けました。残りの10％は公開されないテストセットで、このテストセットのデータにおいてのジャンル予測で競いました。

### 📁 Kalimat Corpus Setup (Manual Step)

The Kalimat Corpus **must be downloaded manually** from:

🔗 [https://sourceforge.net/projects/kalimat/files/kalimat/document-collection/](https://sourceforge.net/projects/kalimat/files/kalimat/document-collection/)

Once downloaded:

1. Extract each ZIP file (like `articlesInternational.zip`, `articlesPolitics.zip`, etc.)
2. Make sure each extracted folder stays separate and is placed directly inside:

```
project-root/
└── data/
    └── KalimatCorpus-2.0/
        ├── articlesInternational/
        │   ├── culturecapr1.txt
        │   ├── culturecapr2.txt
        │   └── ...
        ├── articlesReligion/
        ├── articlesEconomy/
        ├── articlesCulture/
        ├── articlesSports/
        └── articlesLocal/
```

> ✅ Each folder contains `.txt` files — **do not rename or flatten the structure**.

You can verify the structure using exploring_dataset.ipynb:

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--f_lang",
        type=str,
        required=True,
        help="Enter the any Indic Language present in Samanatar Dataset",
    )
    parser.add_argument(
        "--s_lang",
        type=str,
        required=True,
        help="Enter the any Indic Language present in Samanatar Dataset",
    )
    
    LANGUAGES = {"Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
                "Malayalam": "ml", "Marathi": "mr", "Odia": "or", "Punjabi": "pa", "Tamil": "ta", "Telugu": "te", "English": "en"}

    args = parser.parse_args()

    first_language  = args.f_lang
    second_language = args.s_lang
    f_lang = LANGUAGES[first_language]
    s_lang = LANGUAGES[second_language]

    firstSet = f"/data1/home/piyushmishra/newgithub/benchmarks/wat2021-devtest"
    secondSet = f"/data1/home/piyushmishra/newgithub/benchmarks/wat2021-devtest"

    

    # firstSetSI, firstSetSII, secondSetSI, secondSetSII subject to change with a better variable name

    with open(f'{firstSet}/test.en', 'r',encoding='utf-8', errors='ignore') as f:
        firstSetSI = f.readlines()

    with open(f'{firstSet}/test.{f_lang}', 'r',encoding='utf-8', errors='ignore') as f:
        firstSetSII = f.readlines()

    with open(f'{secondSet}/test.en', 'r',encoding='utf-8', errors='ignore') as f:
        secondSetSI = f.readlines()

    with open(f'{secondSet}/test.{s_lang}', 'r',encoding='utf-8', errors='ignore') as f:
        secondSetSII = f.readlines()

    print(f"firstSetSI: {len(firstSetSI)}, Unique firstSetSI: {len(set(firstSetSI))}")
    print(f"firstSetSII: {len(firstSetSII)}, Unique firstSetSII: {len(set(firstSetSII))}")
    print(f"secondSetSI: {len(secondSetSI)}, Unique secondSetSI: {len(set(secondSetSI))}")
    print(f"secondSetSII: {len(secondSetSII)}, Unique secondSetSII: {len(set(secondSetSII))}")

    # print(len(list(set(len(firstSetSI)) & set(len(secondSetSI)))))

    intersect = set(firstSetSI).intersection(secondSetSI)
    # print(intersect[:25])
    print(len(intersect))


    df1 = pd.DataFrame({
        "English":firstSetSI,
        f"{first_language}":firstSetSII
        })
    df2 = pd.DataFrame({
        "English":secondSetSI,
        f"{second_language}":secondSetSII
        })

    df1.English = df1.English.apply(lambda x: x.replace('\n', ''))
    df1[f'{first_language}'] = df1[f'{first_language}'].apply(lambda x: x.replace('\n',''))
    df2.English = df2.English.apply(lambda x: x.replace('\n',''))
    df2[f'{second_language}'] = df2[f'{second_language}'].apply(lambda x: x.replace('\n',''))


    df = pd.merge(df1, df2, on='English')
    print(df.shape)
    df.to_csv(f"/data1/home/piyushmishra/newgithub/test_parallel_data/en{f_lang}{s_lang}_2021_wd.csv",index=False)
    df.drop_duplicates(subset=['English'], inplace=True)
    print(df.shape)
    df.to_csv(f"/data1/home/piyushmishra/newgithub/test_parallel_data/en{f_lang}{s_lang}_2021.csv",index=False)

if __name__ == '__main__':
    main()

import pandas as pd
import random
import string

# Character sets for different languages, separated for weighting
LANGUAGES = {
    "english": {
        "letters": string.ascii_letters,
        "others": string.digits + string.punctuation + " "
    },
    "german": {
        "letters": string.ascii_letters + "äöüÄÖÜß",
        "others": string.digits + string.punctuation + " "
    },
    "spanish": {
        "letters": string.ascii_letters + "ñÑáéíóúÁÉÍÓÚüÜ",
        "others": string.digits + string.punctuation + " "
    },
    "italian": {
        "letters": string.ascii_letters + "àèéìòùÀÈÉÌÒÙ",
        "others": string.digits + string.punctuation + " "
    },
    "greek": {
        "letters": "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω" + "άέήίόύώΆΈΉΊΌΎΏ",
        "others": string.digits + string.punctuation + " "
    },
    "japanese": {
        "letters": "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
        "others": string.digits + string.punctuation + " "
    },
    "russian": {
        "letters": "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
        "others": string.digits + string.punctuation + " "
    },
    "arabic": {
        "letters": "أبتثجحخدذرزسشصضطظعغفقكلمنهوي",
        "others": string.digits + string.punctuation + " "
    },
}

def generate_random_string(language, min_length=5, max_length=20, letter_weight=10):
    """
    Generates a random string of a random length within a given range,
    with weighted character probabilities.
    """
    length = random.randint(min_length, max_length)
    lang_chars = LANGUAGES.get(language, LANGUAGES["english"])
    
    # Create a weighted list of characters
    weighted_char_set = list(lang_chars['letters']) * letter_weight + list(lang_chars['others'])
    
    return ''.join(random.choice(weighted_char_set) for i in range(length))

def generate_random_column_name(min_length=5, max_length=12):
    """Generates a random column name with a random length."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def create_string_dataset(language, num_rows=100):
    """Creates a Pandas DataFrame with random strings and pre/in/suffix columns."""
    base_col_name = generate_random_column_name()
    another_col_name = generate_random_column_name()
    
    data = {
        base_col_name: [generate_random_string(language, 20, 30) for _ in range(num_rows)],
        another_col_name: [generate_random_string(language, 15, 25) for _ in range(num_rows)]
    }
    df = pd.DataFrame(data)

    # Add Prefix, Infix, and Suffix Columns
    base_columns = [base_col_name, another_col_name]

    # Prefix with random length for each row
    prefix_col_name = generate_random_column_name()
    prefix_base_col = random.choice(base_columns)
    df[prefix_col_name] = df[prefix_base_col].apply(lambda x: x[:random.randint(3, 7)])

    # Infix with random length and start for each row
    infix_col_name = generate_random_column_name()
    infix_base_col = random.choice(base_columns)
    def get_random_infix(s):
        if len(s) < 10: # Ensure string is long enough
            return s
        start = random.randint(1, len(s) - 8)
        length = random.randint(3, 7)
        return s[start:start + length]
    df[infix_col_name] = df[infix_base_col].apply(get_random_infix)

    # Suffix with random length for each row
    suffix_col_name = generate_random_column_name()
    suffix_base_col = random.choice(base_columns)
    df[suffix_col_name] = df[suffix_base_col].apply(lambda x: x[-random.randint(3, 7):])

    relationships = {
        "prefix_col": {"name": prefix_col_name, "base": prefix_base_col},
        "infix_col": {"name": infix_col_name, "base": infix_base_col},
        "suffix_col": {"name": suffix_col_name, "base": suffix_base_col}
    }

    return df, relationships

if __name__ == "__main__":
    selected_language = random.choice(list(LANGUAGES.keys()))
    print(f"Generating dataset in: {selected_language.capitalize()}")

    random_df, relationships = create_string_dataset(selected_language, num_rows=100)
    
    print("Column Relationships:")
    for rel_type, rel_info in relationships.items():
        print(f"  - Column '{rel_info['name']}' is a {rel_type.split('_')[0]} of column '{rel_info['base']}'")
    
    # Randomize column order
    shuffled_columns = list(random_df.columns)
    random.shuffle(shuffled_columns)
    random_df_shuffled = random_df[shuffled_columns]

    print("\nDataFrame with Random Column Order:")
    print(random_df_shuffled.head(5))

    print("\nCSV Output (first 5 rows with random column order):")
    print(random_df_shuffled.head(5).to_csv(index=False))
import os
from tkinter import filedialog, Tk, Label, Button, Text, Scrollbar, messagebox




import torch
from fairseq.models.bart import BARTModel

import spacy


def load_text():
    file_name = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_name:
        with open(file_name, "r", encoding="utf-8") as file:
            content = file.read()
        input_text.delete(1.0, "end")
        input_text.insert("end", content)



def save_summary():
    file_name = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_name:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(result_text.get(1.0, "end"))
            messagebox.showinfo("Info", "Summary saved successfully.")




# Load the BART model
bart = BARTModel.from_pretrained('bart.large.xsum', checkpoint_file='model.pt')
bart.to('cpu')
bart.eval()




def summarize_text():
    text = input_text.get(1.0, "end-1c")


    # Split the text into 
    text_parts = []
    text_length = len(text)
    part_length = text_length // 9
    for i in range(9):
        start = i * part_length
        end = (i + 1) * part_length if i < 8 else text_length
        text_parts.append(text[start:end])

    # Use spaCy to extract named entities from the text
    nlp = spacy.load("en_core_web_sm")

    summaries = []
    for part in text_parts:
        doc = nlp(part)
        entities = set([ent.text for ent in doc.ents])

        # Use BART to summarize the text with max_len_b set to 200
        summary = bart.sample(part, beam=4, lenpen=2.0, max_len_b=200, min_len=55, no_repeat_ngram_size=3)

        # Filter out sentences that do not contain named entities
        summary_lines = summary.split('\n')
        filtered_lines = []
        for line in summary_lines:
            if any(entity in line for entity in entities):
                filtered_lines.append(line)
        summary_text = '\n'.join(filtered_lines)
        summaries.append(summary_text)

    # Combine the summaries
    combined_summary = "\n\n".join(summaries)

    # Display the combined summary in the result_text Text widget
    result_text.delete(1.0, "end")
    result_text.insert("end", "Summary:\n\n")
    result_text.insert("end", "\n".join(combined_summary.split("\n")))
    result_text.insert("end", "\n")



root = Tk()
root.title("Text Summarizer")
root.geometry("1200x800")



load_button = Button(root, text="Load Text File", command=load_text, font=("Helvetica", 14), height=2, width=20)
load_button.pack(pady=10)


input_label = Label(root, text="Enter text to be summarized:", font=("Helvetica", 14))
input_label.pack()

input_text = Text(root, wrap="word", height=15, font=("Helvetica", 14))
input_text.pack(pady=10)

summarize_button = Button(root, text="Summarize Text", command=summarize_text, font=("Helvetica", 14), height=2, width=20)
summarize_button.pack(pady=10)

result_label = Label(root, text="Summary:", font=("Helvetica", 14))
result_label.pack()

result_text = Text(root, wrap="word", height=15, font=("Helvetica", 14))
result_text.pack(pady=10)

scrollbar = Scrollbar(root, command=result_text.yview)
scrollbar.pack(side="right", fill="y")



save_button = Button(root, text="Save Summary", command=save_summary, font=("Helvetica", 14), height=2, width=20)
save_button.pack(pady=10)


result_text.config(yscrollcommand=scrollbar.set)

root.mainloop()


from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# initialize splitter object
splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)

# special splitting of code 
code = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

code_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                             chunk_size=50, 
                                                             chunk_overlap=10)

code_chunks = code_splitter.split_text(code)
print('--'*30,'PYTHON_CODE','--'*30)
print(code_chunks)
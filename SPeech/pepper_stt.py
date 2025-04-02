# STT program for MARTY
# This program has pepper listen for words from the user and detects words in a set vocabulary list.
# it assigns the detected word to a specific instruction and asks the user to confirm the operation
# once a directive is confirmed, it is printed to the terminal
from naoqi import ALProxy
import time

# Pepper Robot Connection Info
ip = "192.168.0.109"
port = 9559

# Initialize proxies
audio_proxy = ALProxy("ALSpeechRecognition", ip, port)
tts = ALProxy("ALTextToSpeech", ip, port)
memory = ALProxy("ALMemory", ip, port)

# Set language for speech recognition and text-to-speech
audio_proxy.setLanguage("English")
tts.setLanguage("English")

# Directive state
directive = False
action = "none"

# Define categories and word groups
word_groups = {
    "bakery": ["bread", "cake", "baguette", "croissant", "roll"],
    "fruit": ["apple", "banana", "orange", "tangerine", "mango", "melon", "pineapple", "peach", "plum", "berry", "strawberry", "raspberry", "grape", "cherry"],
    "vegetables": ["onion", "pepper", "lettuce", "cabbage", "carrot", "tomato", "broccoli", "spinach", "kale", "beetroot", "radish", "cauliflower", "celery"],
    "dairy": ["milk", "butter", "margarine", "cheese", "yogurt", "cream"],
    "drinks": ["water", "squash", "juice", "cola", "lemonade", "pepsi", "fanta", "sprite"],
    "alcohol": ["beer", "ale", "wine", "cider", "rum", "whiskey", "brandy", "vodka"],
    "cupboard": ["crisps", "nuts", "seeds", "sweets", "candy", "chocolate", "biscuits", "cookies", "toffee", "snacks"],
    "frozen": ["ice", "pizza", "icecream", "freezer", "chips"],
    "staff": ["manager", "employee", "human", "till", "checkout", "person", "help"],
    "bathroom": ["restroom", "toilets", "washroom"],
    "exit": ["out", "car", "parking", "outside"],
    "police": ["police", "crime", "cops", "sheriff"],
    "ambulance": ["injured", "killed", "hurt", "hospital", "doctor"],
    "fire": ["brigade", "flames", "burning"],
    "no": ["nope", "nah", "not", "don't", "no"],
    "yes": ["yes", "yeah", "yep", "do", "correct"],
    "greetings": ["hello", "hi", "hey", "good", "morning", "evening", "afternoon"]
}

# Flatten vocabulary list
vocabulary = []
for category, words in word_groups.items():
    vocabulary.extend(words)

# Set vocabulary in speech recognition
audio_proxy.setVocabulary(vocabulary, False)

# Start speech recognition
audio_proxy.subscribe("WordRecognizer")
print("[INFO] Pepper is now listening...")

try:
    while True:
        word = memory.getData("WordRecognized")  # Get recognized words
        if word and word[0] != "":  # Ensure it's a valid word
            recognized_word = word[0]  # Extract word
            confidence = word[1]  # Get confidence score

            if confidence > 0.4:
                # Determine the group
                group = None
                for category, words in word_groups.items():
                    if recognized_word in words:
                        group = category
                        break  # Stop once the correct category is found

                # If no group is found
                if group is None:
                    tts.say(f"You said {recognized_word}, but I don't recognize that word.")
                    directive = False
                    action = "none"
                    continue  # Skip the rest of the loop

                print(f"[INFO] Recognized: {recognized_word} (Confidence: {confidence:.2f}) - Category: {group}")

                if not directive:  # If no instruction received prior
                    action = group
                    directive = True

                    if group == "greetings":
                        tts.say("Yes, hello. How can I help?")
                        action = "none"
                        directive = False

                    elif group == "staff":
                        tts.say(f"You said {recognized_word}. Should I find you a member of staff?")
                        
                    elif group == "bathroom":
                        tts.say(f"You said {recognized_word}. Should I show you the bathroom?")
                        
                    elif group == "exit":
                        tts.say(f"You said {recognized_word}. Should I show you the exit?")
                        
                    elif group == "police":
                        tts.say(f"You said {recognized_word}. Should I contact the police department?")
                        
                    elif group == "ambulance":
                        tts.say(f"You said {recognized_word}. Should I call an ambulance?")
                        
                    elif group == "fire":
                        tts.say(f"You said {recognized_word}. Should I contact the fire brigade?")
                    
                    elif group not in ["yes", "no"]:
                        tts.say(f"You said {recognized_word}. Would you like me to show you the {group} section?")
                    
                else:  # Confirm action if instruction received prior
                    if group == "yes":
                        tts.say("Acknowledged")
                        print(f"[INFO] Action confirmed: {action}")

                    elif group == "no":
                        tts.say("Sorry. I misunderstood you")

                    else:
                        tts.say(f"You said {recognized_word}. But I don't understand")

                    action = "none"
                    directive = False

            time.sleep(2)  # Small delay to avoid rapid loop execution

except KeyboardInterrupt:
    print("\n[INFO] Speech recognition stopped.")

finally:
    audio_proxy.unsubscribe("WordRecognizer")
    print("[INFO] Unsubscribed from speech recognition.")

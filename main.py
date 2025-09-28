# main.py (GEPA-only orchestrator, using unified validators.py)

import pprint
from pipeline.executors import passage_executor, questions_executor
from pipeline.new_scoring_method import generate_passage_with_rescoring
import random

executors = {
    "passage": passage_executor,
    "questions": questions_executor
    #"distractors": distractors_executor
}



# --- GEPA Run ---
def run_with_gepa():
    base_prompts = {
        "passage": (
            "SYSTEM: You are an IELTS Academic Reading passage generator.\n\n"
            "TASK:\n"
            "- Write ONE IELTS Academic Reading passage about the given topic.\n"
            "- Length: 800–1000 words (ideal ~900).\n"
            "- Structure: 4–6 paragraphs, each separated by a blank line.\n"
            "- Style: academic, formal, factual, not conversational.\n"
            "- At the very end, include a single line starting with exactly:\n"
            "If you cannot follow ALL requirements, regenerate until you do."
        ),
        "questions": (
            "SYSTEM: You are an IELTS Multiple Choice Question (MCQ) generator.\n\n"
            "TASK:\n"
            "- Write 3 IELTS-style MCQs based on the passage.\n"
            "- Each question must have exactly 4 options (a, b, c, d).\n"
            "- Provide exactly one correct answer.\n"
            "- Output MUST be valid JSON array (and nothing else).\n"
            "- Use this schema:\n\n"
            "[\n"
            "  {\n"
            "    \"id\": \"Q1\",\n"
            "    \"stem\": \"What is the main purpose of ...?\",\n"
            "    \"options\": [\"Option a\", \"Option b\", \"Option c\", \"Option d\"],\n"
            "    \"answer\": \"b\"\n"
            "  },\n"
            "  {\n"
            "    \"id\": \"Q2\",\n"
            "    \"stem\": \"According to the passage, ...\",\n"
            "    \"options\": [\"Option a\", \"Option b\", \"Option c\", \"Option d\"],\n"
            "    \"answer\": \"c\"\n"
            "  },\n"
            "  {\n"
            "    \"id\": \"Q3\",\n"
            "    \"stem\": \"Which of the following is true ...?\",\n"
            "    \"options\": [\"Option a\", \"Option b\", \"Option c\", \"Option d\"],\n"
            "    \"answer\": \"d\"\n"
            "  }\n"
            "]\n\n"
            "STRICT: Output must be valid JSON only, no explanations, no extra text."
        )
    }


    topics = [
    # Science & Technology
    "Artificial Intelligence in Education",
    "The development of the internet",
    "Space exploration and Mars missions",
    "Robotics in everyday life",
    "Nanotechnology innovations",
    "Human cloning debates",
    "The science of renewable energy",
    "Electric vehicles and sustainable transport",
    "Biotechnology and agriculture",
    "3D printing applications",
    "The evolution of smartphones",
    "Quantum computing",
    "Medical imaging technologies",
    "Cybersecurity and data protection",
    "Social media algorithms",
    "Satellite technology",
    "The invention of the steam engine",
    "Renewable materials in construction",
    "The history of aviation",
    "Future of genetic engineering",

    # Environment & Nature
    "Global warming and climate change",
    "The melting of polar ice caps",
    "Deforestation in the Amazon",
    "Endangered species conservation",
    "Coral reefs under threat",
    "Air pollution in urban areas",
    "Plastic waste in oceans",
    "Desertification and drought",
    "Natural disasters and human response",
    "Water scarcity and management",
    "The impact of pesticides on ecosystems",
    "Earthquakes and volcanic activity",
    "Wildfires and forest management",
    "Renewable vs non-renewable resources",
    "Invasive species and ecosystems",
    "The ozone layer recovery",
    "The carbon cycle and climate regulation",
    "Green architecture and eco-buildings",
    "Global fisheries and overfishing",
    "The role of national parks",

    # History & Culture
    "The Silk Road trade routes",
    "The Great Wall of China",
    "The Roman Empire’s engineering feats",
    "The Industrial Revolution",
    "The invention of the printing press",
    "The Renaissance period",
    "Ancient Egyptian civilisation",
    "The history of the Olympic Games",
    "The voyages of Christopher Columbus",
    "The spread of the English language",
    "The Age of Enlightenment",
    "The history of democracy",
    "Vikings and exploration",
    "The French Revolution",
    "The history of photography",
    "Ancient Greek philosophy",
    "The evolution of money and banking",
    "The history of slavery",
    "Castles and medieval society",
    "The history of written scripts",

    # Society & Education
    "Gender equality movements",
    "Literacy and global education",
    "Ageing populations and healthcare",
    "Urbanisation and megacities",
    "Migration and multicultural societies",
    "Child labour issues",
    "Crime prevention methods",
    "Work-life balance",
    "Social media and communication",
    "Mental health awareness",
    "Online learning platforms",
    "The future of higher education",
    "Globalisation and cultural identity",
    "Volunteering and social responsibility",
    "Consumerism and modern lifestyles",
    "The psychology of advertising",
    "Human rights campaigns",
    "Sports and national identity",
    "Food culture and global diets",
    "Fashion and cultural trends",

    # Economy & Development
    "Global trade agreements",
    "The rise of China’s economy",
    "The Great Depression",
    "Tourism and heritage sites",
    "The sharing economy (Uber, Airbnb)",
    "The history of banking",
    "Microfinance in developing nations",
    "International aid and development",
    "The oil industry and geopolitics",
    "The economics of renewable energy",
    "Transportation systems of the future",
    "The space economy and commercialisation",
    "Agricultural revolutions",
    "The digital economy",
    "Global financial crises",
    "International organisations (IMF, WTO)",
    "E-commerce and retail evolution",
    "The gig economy",
    "Poverty reduction strategies",
    "Sustainable development goals (SDGs)",
]

    random_topic = random.choice(topics)

    print("Starting generating")
    generate_passage_with_rescoring(executors, base_prompts, random_topic)



if __name__ == "__main__":
    run_with_gepa()
    
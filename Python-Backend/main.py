#  Add Embeddings to enhance quality of hints and scoring. 
#  Add a way to score the thought process (clear methodical thinking or all over the place)
#  Include adaptive difficulty 
#  Incorporate Time limits into final score calculation (like F1 time punishments)
#  Use RLHF Somehow? 
#  Include Voice integration (Whisper or Google Speech to Text)
#  Add Text-to_Speech (CoquiTTS, ElvenLabs, Amazon Polly, Google Text-to-Speech, Azure Text-to-Speech, IBM Watson Text-to-Speech, etc.)
#  Include Phone Call integration using VOCODER? Then save latex file of score and evaluation to Google Drive with interviewee's name and information. 
#  With multiple interviewees, rank them using the ELO algorithm with further interview phases competing similar ELO users against each other at the same difficulty levels.
#  Incorporate LangChain
#  Connect Alfred AI Assistant Voice Program to this program
#  Ensure Questions/Answer/Grading Rubrics are fully generated at start in a form of JSON file and split into individual questions 
#       Make sure the grading rubric is broken down properly. 
#       Use GPT-3.5 Turbo for Q&A and quick conversational responses, then use GPT-4 for the technical aspect and grading. 
#       This can be done in parallel with the emotional interviewing process using multiple threads
#       After Splitting up questions, answers and rubrics can all be computed simultaneously using multiple threads as they are graded independently
#       Once these have been generated, a flag is set to begin the technical interview. 
#       After each question is completed, it must be graded and scored in a seperate thread while the interview continues.
#       At the end of the technical interview, a short Q&A session is to be held with the interviewee to determine their interest in the position and any questions they may about the company
#       Employ streaming of whisper, ChatCompletion, and TTS to create a more natural interview experience
#       Employ soft and hard cut-off-times for each question to ensure the interviewee is not taking too long to answer a question (do not consider time in scoring)
#       Submit full report in LaTex at the end, sent to pdf file. 
#  Incorporte real recent events into discussion -> Langchain search tool.

# 1) Fix Global and Short Term Memory Problems (Individual Question feedback should only consider the user's thought process for 
#    that question while wholistic feedback should consider the user's thought process for all questions.) Both Memory types should
#    know what the AI assistant is saying as well. Print JSON of memories in txt file at the end of interview


import openai, os, json, re

openai.api_key = os.environ["OPENAI_API_KEY"]

global_memory = []
short_term_memory = ""
hints_requested = []
scores = []
user_responses = []

Industry = "Hedge Fund"
Position = "Quantitative Analyst"

def call_gpt(prompt, user_messages=[], model="gpt-4"):
    messages = [
        {"role": "system", "content": f"""You are a top-level {Industry} interviewer. You are interviewing a candidate for the position of {Position}. 
        Consider this prompt: """},
    ]
    
    for message in user_messages:
        messages.append({"role": "user", "content": message})

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.8,
    )

    message = response['choices'][0]['message']['content']
    return message.strip()

def generate_question(difficulty):
    """
    Generate a complex brain teaser based on the given difficulty level.

    Args:
        difficulty (int): The difficulty level of the brain teaser.

    Returns:
        str: A generated brain teaser.
    """
        
    question_prompt = f"""Generate a brain teaser relating to one of these topics: (logic, puzzles, probability, statistics, financial, trading, game theory, estimation, mathematics, or physics) 
    for a top-level hedge fund interview with a difficulty of {difficulty}. Choose one of the presented topics randomly. Do not include any hints in the question. 
    Do not provide a solution. State the difficulty of the question as well as the topic. The problem must not use excessively technical jargon while remaining highly difficult."""

    question = call_gpt(question_prompt)
    global_memory.append(question)
    return question

def generate_hint(question, short_term_memory):

    hint_prompt = f"""Given the user's thought process: ('{short_term_memory}'), provide a hint for the question: {question}. 
                    If the thought process is in the right direction, confirm that it is in the right direction and provide a hint that will help the user get closer to the solution. 
                    If the thought process is in the wrong direction, state that the logic is flawed as well as how it is flawed and provide a hint that will help the user get closer to the right direction.
                    If there is no thought process, give a simple hint that will help the user get started. 
                    Do not provide the complete solution as part of the hint. Be completely honest, critical, and insightful about the user's thought process. """

    return call_gpt(hint_prompt)

def score_solution(question, thought_process):
    # user_messages = [question]
    # user_messages.extend(global_memory)

    scoring_template_prompt = f"""Generate a highly critical and methodical scoring template specific to this brain teaser: {question}. 
    It must be able to detect false information and misleading thought processes. It must also give points for a valid thought process that is in the right direction. 
    The total score should sum to 10."""

    scoring_template = call_gpt(scoring_template_prompt)
    print(scoring_template)

    scoring_prompt = f""" 
    Use this scoring template: {scoring_template}, for the question {question}.
    The interviewee will often give false information or invent a thought process to get a higher score.
    If no valid thought process is provided or the information is completely irrelevant, the score should be 0.
    Now score the interviewee's solution: {thought_process} using the template specified above.
    Give a score value out of 10 in the format [score/10].  
    """
    score = call_gpt(scoring_prompt)
    print(score)

    match = re.search(r'(\d+(\.\d+)?)/10', score)
    if match:
        score = match.group(1)

    try:
        # Handle the case when the score is returned as a fraction
        if '/' in score:
            numerator, denominator = score.split('/')
            score = float(numerator) / float(denominator) * 10

        return float(score)


# Fix this to Handle the case when the score is returned within a sentence and not alone. 
    except ValueError:
        print(f"Unable to score the solution: {score}")
        return 0



def interview(n):

    global short_term_memory

    for i in range(n):
        difficulty = i + 1
        question = generate_question(difficulty)
        print(f"{i + 1}) {question} [Difficulty: {difficulty}]")

        hints_used = 0
        user_input = ""
        while user_input.lower() != "done":
            user_input = input("Enter your thought process, 'help' for a hint, or 'done': ")

            if user_input.lower() == "help":
                hints_used += 1
                hint = generate_hint(question, short_term_memory)
                print(f"Hint: {hint}")

            elif user_input.lower() != "done":
                short_term_memory += user_input + " "
                global_memory.append(user_input)
                user_responses.append(user_input) # Add this line

        score = score_solution(question, short_term_memory)
        scores.append(score)
        hints_requested.append(hints_used)
        short_term_memory = ""

    return sum(scores)

def analyze_performance(global_memory, hints_requested, scores):
    questions_and_answers = ', '.join(global_memory)
    hints_requested_str = ', '.join(map(str, hints_requested))
    scores_str = ', '.join(map(str, scores))

    prompt = f"Analyze the user's performance in a top-level hedge fund interview based on the questions the they were asked and asnwers they gave: '{questions_and_answers}'. Examine the problems along with the optimal solution and compare that to the interviewee's solution. Analyze their overall solutions, their logic and thought processes, and provide insightful information on the user's responses to these questions. Provide a summary of their strengths and weaknesses for each question. Be highly critical of any mistakes made."
    
    analysis = call_gpt(prompt)
    return analysis


def main():
    num_questions = int(input("Enter the number of questions for the interview: "))
    final_score = interview(num_questions)
    print(f"Your total score is: {final_score} out of {num_questions * 10}")

    # Analyze the user's performance
    performance_analysis = analyze_performance(global_memory, hints_requested, scores)
    print("\nPerformance Analysis:")
    print(performance_analysis)

    print("\nGlobal Memory:" + str(global_memory))

if __name__ == "__main__":
    main()
 

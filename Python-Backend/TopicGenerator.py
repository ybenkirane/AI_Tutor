import random,json
from interview import gpt_Q_Gen

def get_topic(topic):
    topic_format = """
    {
        "Supertopic": {
            "Subtopic L1": {
                "Subtopic L2": {
                    "Subtopic L3": {},
                    "Subtopic L3": {},
                    etc...
            },
                "Subtopic L2": {
                    "Subtopic L3": {},
                    "Subtopic L3": {},
                    etc...
                }, 
                etc...
            },
            etc...
        }
    }
    """
    
    topic_prompt = f""" Given a general supertopic, create a flow chart of every sub-topic down to 3 levels of depths associated to a certain topic. The super-topic is {topic}. Each Level should have roughly 3 sub-topics. Follow this format:  """ + topic_format
    gpt_Q_Gen(topic_prompt, title="topic_json", model="gpt-4")


def random_branch(json_obj):
    if not isinstance(json_obj, dict) or not json_obj:
        return []

    key = random.choice(list(json_obj.keys()))
    return [key] + random_branch(json_obj[key])

def random_branches(json_obj, count):
    branches = []
    for _ in range(count):
        branches.append(random_branch(json_obj))
    return branches

def pretty_print_branches(branches):
    topic_list = []
    for branch in branches:
        topic_list.append(" > ".join(branch))
        print(topic_list[-1])
    return topic_list

def topic_generator(topic, num_branches):
    get_topic(topic)
    with open("topic_json.json") as f:
        topic_json = json.load(f)

    # Get and print 3 random branches
    branches = random_branches(topic_json, num_branches)
    return pretty_print_branches(branches)


if __name__ == "__main__":
    topic_generator("Brain Teasers for Quantitative Finance", 5)

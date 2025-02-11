from dataclasses import dataclass
from typing import Dict


@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str
    # stop_token_id: int


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )


template_dict: Dict[str, Template] = dict()

register_template(
    template_name='chatglm3',
    system_format='{content}',
    user_format='{content}',
    assistant_format='{content}',
    system="You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
    stop_word='</s>',
)

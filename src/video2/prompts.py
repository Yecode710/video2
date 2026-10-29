"""Fixed instructions: no demonstrations, transcripts, or reference summaries."""


def make_prompt(language: str, max_words: int) -> str:
    if max_words < 1:
        raise ValueError("summary length must be positive")
    if language == "en":
        return (
            f"Summarize this video in English in at most {max_words} words. "
            "Describe the main events, actions, and important visible details in temporal order. "
            "Use only evidence visible in the video; do not invent speech, sounds, identities, "
            "motives, or events. Return only the summary, without a title or bullet points."
        )
    if language == "zh":
        return (
            f"请用不超过{max_words}个汉字概括这个视频。"
            "按时间顺序描述主要事件、动作和重要的可见细节。"
            "只依据视频中可见的内容，不要编造对话、声音、人物身份、动机或事件。"
            "仅输出摘要，不要标题或项目符号。"
        )
    raise ValueError("language must be en or zh")

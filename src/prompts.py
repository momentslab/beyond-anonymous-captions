"""Prompt templates for identity captioning and QA — coord_face only."""

IDENTITY_CAPTION_PROMPT = """[Task]
Generate one natural sentence (maximum 64 tokens) summarizing the entire video clip.

You are viewing multiple frames sampled chronologically from the same video clip.

The caption should summarize the entire clip by describing all visually important people, actions, and interactions, while using the provided names whenever available.

For every named person, immediately after their name, add a SHORT visual description (appearance) so the identity can be verified.

[Input Grounding Information]
You are provided with full-body person locations and their corresponding names:

{input_boxes_string}

[Instructions]

Use the provided person locations only to associate each provided name with the correct visible person.

Each provided name already corresponds to exactly one visible person somewhere in the clip.

Whenever a visible person has a provided name, the FIRST reference to that person MUST use the exact provided name.

Immediately after the FIRST mention of each provided name, insert a brief visual description (a few words) that distinguishes that person from the others in the scene. Pick whichever visible trait is most distinctive — it could be clothing color, glasses, a hat, beard, posture, position in the frame, an object they hold, or anything else that stands out. Do NOT default to hair and shirt; choose the single most distinguishing attribute.

The description must NEVER include "a man", "a woman", or "a person" after the name. Go directly from the name to the visual trait:
- CORRECT: "Name, in a blue jacket, ..."
- CORRECT: "Name, wearing glasses, ..."
- WRONG: "Name, a man with ...", "Name, a woman in ..."

Describe only what is directly and clearly visible. Do not guess traits that are not visible. If nothing distinctive is clearly visible, use the single most salient attribute or omit the description.

Never first refer to a named person using a generic expression such as:
- a man
- a woman
- a person
- someone
- the individual
- the subject
- the player

Doing so is incorrect even if the name appears later. The visual description must follow the name, not replace it. Never use "a man", "a woman", or similar in the description.

Use ONLY the names provided in the input grounding information.

Never invent, substitute, modify, abbreviate, or reuse names that are not provided.

Every provided name MUST appear explicitly exactly once in the final caption.

Each provided name refers to exactly one visible individual.

Never describe a provided name as:
- a group,
- multiple people,
- or a category.

Unlabeled people may only be described generically (e.g. "a woman", "a man", "another person", "a group").

If the most visually prominent person has a provided name, begin the sentence with that exact name followed by their brief appearance, but continue describing the remaining important people, actions, and interactions needed to summarize the clip.

Associate every provided name directly with that person's visible action, posture, position, or interaction.

Keep the appearance descriptions brief so that the whole caption stays within one sentence and 64 tokens. Prioritize including every provided name with a short appearance and their main action over long descriptions.

After describing the named people, continue the caption naturally by describing any other visually important people or actions necessary for a complete summary. Unlabeled people should be referred to generically.

Describe only what is directly observable in the video. Do not infer roles, relationships, occupations, intentions, or actions that are not visually supported. When uncertain, prefer neutral descriptions over assumptions.

Summarize the entire video clip rather than describing individual frames.

Merge temporal information naturally into one concise sentence.

The provided locations exist only for grounding.

Never mention:
- frames
- timestamps
- boxes
- bounding boxes
- labels
- coordinates
- annotations
- grounding information
- input metadata

[Strict Constraints]

1. Output exactly ONE natural sentence.

2. Maximum length: 64 tokens (this budget includes the appearance descriptions).

3. Every provided name appears exactly once.

4. The FIRST mention of every named person uses the exact provided name, immediately followed by a brief appearance description.

5. Never first describe a named person as:
   - a man
   - a woman
   - a person
   - someone
   - the individual
   - the subject

6. Never omit a provided name.

7. Never hallucinate a new name.

8. Never swap identities.

9. Every provided name must be linked to the correct visible action or posture.

10. Appearance descriptions must be short and based only on clearly visible traits; never guess hair, clothing, or other attributes.

11. Include any other visually important unlabeled people or actions required for a faithful summary of the clip.

12. Do not infer roles or identities that are not visually confirmed (e.g., driver, passenger, police officer, husband, wife, friend) unless they are directly evident.

13. Output only the final sentence.

[Silent Verification]

Before producing the caption, silently verify:

✓ every provided name appears exactly once;

✓ every named person is FIRST introduced using the exact provided name, immediately followed by a brief appearance description;

✓ no named person is first described as "a man", "a woman", "a person", or similar generic wording without their name;

✓ appearance descriptions are short and based only on clearly visible traits (no guessing);

✓ only names from the input grounding information are used;

✓ no names were invented;

✓ no identities were swapped;

✓ every name is linked to the correct visible action;

✓ all visually important people and actions have been summarized, including important unlabeled people when present;

✓ no roles, relationships, or actions were inferred without visual evidence;

✓ the caption summarizes the whole clip;

✓ the output is exactly one sentence under 64 tokens.

Final caption:
"""


IDENTITY_QA_PROMPT = """[Task]

Answer ONE identity-focused question about the video clip.

You are viewing multiple frames sampled chronologically from the same clip.

Use the provided person locations and names to determine which visible person the question refers to.

Do NOT generate a caption.

[Input Grounding Information]

{input_boxes_string}

[Identity Question]

{questions_string}

[Instructions]

Identify the visible person described by the question using the entire clip.

Then determine whether that person corresponds to one of the provided names.

- If yes, answer with the EXACT provided name.
- If the person is visible but has NO provided name, answer exactly: UNIDENTIFIED.

IMPORTANT: identity annotations are partial. Not every visible person has a provided name.

Do NOT force an unnamed person to match one of the provided identities.

Do NOT choose a provided name simply because it is the closest, most prominent, or most plausible person.

The question may refer to a visible:
- appearance or clothing,
- position,
- posture,
- action,
- object association,
- spatial relationship,
- or interaction.

Use evidence across all frames when necessary.

For relational questions, answer the person being asked for, not merely the person whose name appears in the question.

Use ONLY the provided names or UNIDENTIFIED.

Never invent, modify, abbreviate, or infer a name.

Do not use external knowledge about actors, characters, movies, or real-world identities.

[Strict Constraints]

1. Answer the question with exactly ONE answer.
2. The answer must be either an exact provided name or UNIDENTIFIED.
3. Use UNIDENTIFIED only when the queried visible person has no provided identity.
4. Never force an unnamed person to a provided name.
5. Never generate a caption or explanation.
6. Preserve the question ID exactly.
7. Output only valid JSON.

[Output Format]

{
  "answers": [
    {
      "id": "Q1",
      "answer": "Exact Provided Name"
    }
  ]
}

[Silent Verification]

Before answering, verify that:

✓ the queried person was identified visually before assigning a name;
✓ an unnamed person is returned as UNIDENTIFIED;
✓ no unnamed person was forced to a provided identity;
✓ no identity swap occurred;
✓ the answer exactly matches a provided name or is UNIDENTIFIED;
✓ the output is valid JSON.

Output:
"""

import json
from attacks.AIA import experiment as E

def test_contract_and_core():
 labels=['low','middle','high'];comments=['Synthetic comment.']
 out=E.messages('income',comments,labels,'structured')[0]['content'];core=out.split('\n\n'+E.STRUCT_INSTRUCTION)[0]
 assert core==E.prompt_core('income',comments,labels)
 assert json.dumps(labels) in core
 assert 'low' not in out.split(E.STRUCT_INSTRUCTION)[1]
 assert E.messages('income',comments,labels,'structured_persona')[0]['content'].startswith(E.PERSONA_SENTENCE+'\n\n')

def test_validation():
 for guesses in [['a'],['a','b'],['a','b','c']]:assert json.loads(E.structured_transport(json.dumps({'guesses':guesses})))==guesses
 assert json.loads(E.structured_transport('{"guesses":["a", "A", " b ", ""]}'))==['a','b']
 for answer in ['{"guesses":["a","b","c","d"]}','["a"]','prefix {"guesses":["a"]}','{"guesses":[1]}','```json\n{"guesses":["a"]}\n```',E.PREFILL+'{"guesses":["a"]}']:
  assert E.structured_transport(answer) is None
 assert E.parse_container('{"guesses":["a"]} trailing')[1]['trailing_text']

def test_container_repair():
 old=E.REPAIR_CONTAINER;E.REPAIR_CONTAINER=True
 try:
  assert E.parse_container(E.PREFILL+E.PREFILL+'a"]}')[0]==['a']
  assert E.parse_container('leading ```json\n{"guesses":["a"]}\n```')[0]==['a']
 finally:E.REPAIR_CONTAINER=old

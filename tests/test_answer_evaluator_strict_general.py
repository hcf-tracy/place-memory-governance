from hf13.answer_evaluator_strict_general import (
    EVALUATOR_VERSION, scalar_match, structured_expected_match, score_answer_strict_general
)

def test_v32_is_public_primary_evaluator():
    assert EVALUATOR_VERSION == '3.2-strict-general'

def test_representation_matching():
    assert scalar_match('DRY HUMOR', 'dry_humor')
    assert scalar_match('The reading is 24,680.', '24680')
    assert scalar_match('Friday at 9:30 pm', 'Friday 21:30')
    assert not scalar_match('twenty-four', '24')

def test_only_declared_aliases_are_accepted():
    fact={'entity':'object','predicate':'status','value':'available','value_aliases':['ready']}
    assert structured_expected_match('The object is ready.',fact)
    gold={'turn_id':'demo-1','query_semantics':{'answer_obligations':[{'type':'values_all','values':['ready']}]},'expected_facts':[],'ineligible_facts':[]}
    r=score_answer_strict_general('It is ready.',gold,'What is its status?')
    assert r.evaluable and r.answer_f1==1.0
    r=score_answer_strict_general('It is offline.',gold,'What is its status?')
    assert r.evaluable and r.answer_f1==0.0

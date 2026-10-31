"""Check local Full LangGraph parity with the team's CRAG class without model calls."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import run_live as experiment
from rag.pipelines.crag import CRAG
from rag.pipelines import prompts
from rag.types import Document, QAExample
from rag.utils.timing import StageTimer

class Retriever:
    def __init__(self,docs):self.docs=docs
    def retrieve(self,q,k):return self.docs
    def search(self,q,k):return self.docs
class LLM:
    def __init__(self,responses):self.responses=responses;self.seen=[]
    def generate(self,prompt):self.seen.append(prompt);return self.responses[prompt]
class Backend:
    def __init__(self,llm):self.llm=llm
    def request(self,kind,prompt,fallback,calls):return self.llm.generate(prompt)

def main():
    docs=[Document(id=str(i),text='doc'+str(i),score=.9-i*.1) for i in range(5)]
    question='Question?';retriever=Retriever(docs)
    for scores in [[.8,.1,.9,.2,.5],[.1,.2,.0,.1,.2],[.4,.2,.1,.5,.2]]:
        action='correct' if max(scores)>=.7 else ('incorrect' if max(scores)<.3 else 'ambiguous')
        contexts=[d for d,v in zip(docs,scores) if v>=(.7 if action=='correct' else .3)]
        if action=='incorrect':contexts=docs
        elif action=='ambiguous':contexts+=docs
        responses={prompts.GRADE.format(question=question,document=d.text):str(v) for d,v in zip(docs,scores)}
        responses[prompts.REWRITE.format(question=question)]='rewritten'
        responses[prompts.GENERATE.format(context=prompts.format_context(contexts),question=question)]='answer'
        old_llm=LLM(responses);new_llm=LLM(responses)
        baseline=CRAG(retriever,old_llm,retriever).run(QAExample(id='q',question=question,answers=['answer']))
        graph=experiment.make_graph(Backend(new_llm),retriever,.7,.3,True)
        result=graph.invoke({'row':{'id':'q','question':question},'variant':'full','scores':[],'index':0,'timer':StageTimer(),'calls':[]})
        assert result['action']==baseline.corrective_action
        assert result['answer']==baseline.answer
        assert [d.id for d in result['contexts']]==[d.id for d in baseline.contexts]
        assert result['scores']==baseline.grader_scores
        assert new_llm.seen==old_llm.seen
    print('PASS: local Full graph matches team CRAG prompts, call order, scores, actions, contexts and answers in all three branches. No model calls.')
if __name__=='__main__':main()

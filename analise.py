import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import json

load_dotenv()

class SentimentAnalyzer:
    @staticmethod
    def is_local_available():
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def __init__(self, engine="Gemini"):
        """
        Inicializa o analisador de sentimentos com o motor escolhido.
        Motores suportados: "Gemini", "OpenAI", "Local".
        """
        self.engine = engine
        
        if self.engine == "Gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY não encontrada no arquivo .env")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
        elif self.engine == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini" # Econômico e eficiente
            
        elif self.engine == "Local":
            try:
                import torch
                from transformers import pipeline, AutoTokenizer
                self.model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.classifier = pipeline("sentiment-analysis", model=self.model_name, tokenizer=self.tokenizer)
            except ImportError:
                raise ImportError("Motor local indisponível no Python 3.13. Use Gemini ou OpenAI.")

    def analyze(self, text):
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRO", "score": 1.0, "details": "Texto vazio"}

        prompt = f"""
        Analise o sentimento do seguinte texto em português. 
        Responda APENAS com um formato JSON válido:
        {{"label": "POSITIVO" ou "NEGATIVO" ou "NEUTRO", "score": float entre 0 e 1, "explanation": "breve explicação"}}
        
        Texto: "{text}"
        """

        if self.engine == "Gemini":
            response = self.model.generate_content(prompt)
            try:
                content = response.text.strip().replace('```json', '').replace('```', '')
                result = json.loads(content)
                usage = getattr(response, 'usage_metadata', None)
                if usage:
                    result['usage'] = {"prompt_tokens": usage.prompt_token_count, "candidates_tokens": usage.candidates_token_count, "total_tokens": usage.total_token_count}
                return result
            except: return {"label": "ERRO", "score": 0}

        elif self.engine == "OpenAI":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            try:
                result = json.loads(response.choices[0].message.content)
                usage = response.usage
                result['usage'] = {"prompt_tokens": usage.prompt_tokens, "candidates_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                return result
            except: return {"label": "ERRO", "score": 0}

        else: # Local
            res = self.classifier(text)[0]
            mapping = {"LABEL_0": "NEGATIVO", "LABEL_1": "NEUTRO", "LABEL_2": "POSITIVO"}
            return {"label": mapping.get(res['label'], res['label']), "score": res['score']}

    def extract_topics(self, texts):
        if self.engine == "Local":
            return {"text": "Extração de tópicos requer um motor de IA (Gemini ou OpenAI)."}
            
        combined_text = "\n---\n".join(texts[:15])
        prompt = f"""
        Analise as seguintes menções sobre uma empresa e identifique os 3 principais tópicos, o sentimento predominante e uma ação sugerida.
        Responda em Português do Brasil.
        
        "{combined_text}"
        """

        if self.engine == "Gemini":
            response = self.model.generate_content(prompt)
            usage = getattr(response, 'usage_metadata', None)
            return {"text": response.text, "usage": {"prompt_tokens": usage.prompt_token_count if usage else 0, "candidates_tokens": usage.candidates_token_count if usage else 0, "total_tokens": usage.total_token_count if usage else 0}}
        
        elif self.engine == "OpenAI":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            usage = response.usage
            return {"text": response.choices[0].message.content, "usage": {"prompt_tokens": usage.prompt_tokens, "candidates_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}}

class SocialMonitor:
    def __init__(self):
        self.ddgs = DDGS()

    def search_mentions(self, company, platform, max_results=20):
        # Estratégia 1: Busca restritiva (mais precisa)
        query_site = f'"{company}" site:{platform}.com'
        if platform == "twitter": query_site = f'"{company}" (site:twitter.com OR site:x.com)'
        
        # Estratégia 2: Busca ampla (fallback)
        query_broad = f'"{company}" {platform} news comments'
        
        try:
            # Tentar busca restritiva
            results = self.ddgs.text(query_site, max_results=max_results)
            
            # Se não houver resultados suficientes, tentar a ampla e mesclar
            if len(results) < 5:
                broad_results = self.ddgs.text(query_broad, max_results=max_results)
                results.extend(broad_results)
            
            # Limpar duplicados e formatar
            seen = set()
            unique_results = []
            for r in results:
                if r['href'] not in seen:
                    unique_results.append({
                        "texto": r.get('body', r.get('title', '')), 
                        "fonte": r['href'], 
                        "titulo": r['title']
                    })
                    seen.add(r['href'])
            return unique_results[:max_results]
        except Exception as e:
            print(f"Erro na busca do Radar ({platform}): {e}")
            return []

    def analyze_social_trends(self, mentions, analyzer: SentimentAnalyzer):
        """
        Analisa as menções para gerar os 20 principais assuntos em formato de tabela.
        """
        if not mentions: return []
        
        combined_text = "\n---\n".join([m['texto'] for m in mentions[:30]])
        prompt = f"""
        Com base nestas menções: "{combined_text}"
        Classifique os 20 principais assuntos sobre a empresa.
        Responda APENAS JSON: {{"temas": [{{"topico": "Assunto", "sentimento": "Positivo/Negativo/Neutro", "observacao": "Breve texto"}}]}}
        """
        
        import re
        try:
            if analyzer.engine == "Gemini":
                res = analyzer.model.generate_content(prompt)
                text = res.text
            elif analyzer.engine == "OpenAI":
                res = analyzer.client.chat.completions.create(
                    model=analyzer.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                text = res.choices[0].message.content
            
            # Extrair JSON usando Regex
            json_match = re.search(r'(\{.*\})', text.replace('\n', ' '), re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get('temas', [])
        except Exception as e:
            print(f"Erro no Radar: {e}")
            return []
        return []
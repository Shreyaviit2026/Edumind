
import os
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from services.embedding_service import get_embedding_service
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGService:

    
    def __init__(self):
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Remove quotes if present
        groq_api_key = groq_api_key.strip('"').strip("'")
        
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",  # Updated to current supported model
            temperature=0.7
        )
    
    def generate_summary(self, subject: str, unit: str, chapter: Optional[str] = None) -> Dict:

        embedding_service = get_embedding_service()
        
        # Get all documents content
        content = embedding_service.get_all_documents_content(subject, unit)
        
        if not content:
            return {
                "status": "error",
                "message": "No content found for this subject/unit"
            }
        
        # Limit content size (first 15000 characters to avoid token limits)
        if len(content) > 15000:
            content = content[:15000] + "..."
        
        # Summary prompt
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""You are an expert educator. Summarize the following educational content into a structured, note-wise format with clear bullet points.

Content:
{content}

Create a comprehensive summary with:
1. Main topics and concepts
2. Key definitions and terminology
3. Important formulas or principles (if applicable)
4. Examples and applications

Format the summary with clear headings and bullet points for easy study."""
        )
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"content": content})
            
            return {
                "status": "success",
                "subject": subject,
                "unit": unit,
                "summary": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating summary: {str(e)}"
            }
    
    def generate_mcqs(self, subject: str, unit: str, count: int = 10, previous_questions: list = None) -> Dict:

        embedding_service = get_embedding_service()
        
        # Get all documents content
        content = embedding_service.get_all_documents_content(subject, unit)
        
        if not content:
            return {
                "status": "error",
                "message": "No content found for this subject/unit"
            }
        
        # Limit content size
        if len(content) > 12000:
            content = content[:12000] + "..."
        
        # Build previous questions context
        previous_context = ""
        if previous_questions and len(previous_questions) > 0:
            previous_context = "\n\nPREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT OR CREATE SIMILAR QUESTIONS):\n"
            for i, pq in enumerate(previous_questions, 1):
                previous_context += f"{i}. {pq}\n"
            previous_context += "\nIMPORTANT: Create questions on DIFFERENT topics/concepts than the above questions.\n"
        
        # MCQ prompt
        prompt = PromptTemplate(
            input_variables=["content", "count", "previous_context"],
            template="""You are an expert educator creating diverse multiple choice questions. Based on the following educational content, create {count} multiple choice questions that cover DIFFERENT topics and concepts from across the entire content.

Content:
{content}
{previous_context}
IMPORTANT INSTRUCTIONS:
1. Cover DIVERSE topics - each question should test a different concept or area from the content
2. If previous questions are provided above, ensure your new questions cover COMPLETELY DIFFERENT topics
3. Vary difficulty levels - include easy, medium, and challenging questions
4. Use different question types:
   - Factual recall ("What is...?", "Which of the following...?")
   - Conceptual understanding ("Why does...?", "How does...?")
   - Application-based ("In which scenario...?", "What would happen if...?")
   - Analysis ("Compare...", "What is the relationship between...?")
5. Ensure NO repetitive or similar questions
6. Make all distractors (wrong options) plausible but clearly incorrect

For each question, provide:
1. The question text
2. Four options (A, B, C, D)
3. The correct answer (A, B, C, or D)
4. A brief explanation

Format each MCQ as follows:
Question X: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]
Explanation: [brief explanation]

---

Create {count} DIVERSE questions covering DIFFERENT concepts and topics from the entire content."""
        )
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"content": content, "count": count, "previous_context": previous_context})
            
            # Parse MCQs
            mcqs = self._parse_mcqs(result)
            
            return {
                "status": "success",
                "subject": subject,
                "unit": unit,
                "count": len(mcqs),
                "mcqs": mcqs
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating MCQs: {str(e)}"
            }
    
    def _parse_mcqs(self, mcq_text: str) -> List[Dict]:

        mcqs = []
        
        # Clean up markdown formatting
        cleaned = mcq_text.replace('**', '').replace('*', '')
        
        # Try splitting by various separators
        # First try "---" separator
        questions = re.split(r'\n\s*---+\s*\n', cleaned)
        
        # If only one block, try splitting by "Question X" pattern
        if len(questions) <= 1:
            questions = re.split(r'\n(?=(?:Question\s*\d+[.:]))', cleaned)
        
        # If still only one block, try splitting by question blocks that 
        # follow "Correct answer:" lines (LLM conversational format)
        if len(questions) <= 1:
            # Split on patterns like "Here is the new question:" or blank lines before a question
            parts = re.split(r'(?:Let me know[^\n]*\n+(?:Yes[^\n]*\n+)?(?:.*?new question[^\n]*\n)?)', cleaned, flags=re.IGNORECASE)
            if len(parts) > 1:
                questions = parts
        
        # If still only one block, try splitting by numbered pattern like "1." or "1)"
        if len(questions) <= 1:
            questions = re.split(r'\n(?=\d+[.)]\s+)', cleaned)
        
        for q_text in questions:
            if not q_text.strip():
                continue
            
            try:
                mcq = self._parse_single_mcq(q_text)
                if mcq:
                    mcqs.append(mcq)
            except Exception:
                continue
        
        return mcqs
    
    def _parse_single_mcq(self, q_text: str) -> Optional[Dict]:
        """Parse a single MCQ from text with flexible format handling."""
        
        lines = q_text.strip().split('\n')
        if len(lines) < 5:  # Need at least question + 4 options
            return None
        
        # Strategy 1: Try structured regex parsing
        result = self._try_regex_parse(q_text)
        if result:
            return result
        
        # Strategy 2: Try line-by-line parsing
        result = self._try_line_parse(lines)
        if result:
            return result
        
        return None
    
    def _try_regex_parse(self, q_text: str) -> Optional[Dict]:
        """Try to parse MCQ using flexible regex patterns."""
        
        # Extract question - try multiple patterns
        question = None
        question_patterns = [
            r'Question\s*\d*[.:]*\s*(.+?)(?=\n\s*[A-Da-d][\).:])',  # Question X: ... \n A)
            r'^\d+[.)]\s*(.+?)(?=\n\s*[A-Da-d][\).:])',              # 1. ... \n A)
            r'^(.+?)(?=\n\s*[A-Da-d][\).:])',                         # First line before options
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, q_text, re.DOTALL | re.MULTILINE)
            if match:
                question = match.group(1).strip()
                if len(question) > 10:  # Reasonable question length
                    break
                question = None
        
        if not question:
            return None
        
        # Extract options - try multiple formats: A) A. A: a)
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            option_patterns = [
                rf'(?:^|\n)\s*{letter}[\).:\s]\s*(.+?)(?=\n\s*[B-Eb-e][\).:\s]|\nCorrect|\nAnswer|\nExplanation|\n\s*$|\Z)',
                rf'(?:^|\n)\s*{letter.lower()}[\).:\s]\s*(.+?)(?=\n\s*[b-eb-e][\).:\s]|\nCorrect|\nAnswer|\nExplanation|\n\s*$|\Z)',
            ]
            for pattern in option_patterns:
                match = re.search(pattern, q_text, re.DOTALL)
                if match:
                    opt_text = match.group(1).strip().split('\n')[0].strip()  # Take first line only
                    if opt_text:
                        options[letter] = opt_text
                        break
        
        if len(options) < 4:
            return None
        
        # Extract correct answer
        correct_answer = "A"
        answer_patterns = [
            r'Correct\s*[Aa]nswer\s*[.:]\s*([A-Da-d])(?:\s*[\).])?',
            r'[Aa]nswer\s*[.:]\s*([A-Da-d])(?:\s*[\).])?',
            r'Correct\s*[.:]\s*([A-Da-d])(?:\s*[\).])?',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, q_text, re.IGNORECASE)
            if match:
                correct_answer = match.group(1).upper()
                break
        
        # Extract explanation
        explanation = ""
        explanation_patterns = [
            r'Explanation\s*[.:]\s*(.+?)(?=\n---|$)',
            r'Reason\s*[.:]\s*(.+?)(?=\n---|$)',
        ]
        for pattern in explanation_patterns:
            match = re.search(pattern, q_text, re.DOTALL | re.IGNORECASE)
            if match:
                explanation = match.group(1).strip()
                break
        
        return {
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation
        }
    
    def _try_line_parse(self, lines: List[str]) -> Optional[Dict]:
        """Try to parse MCQ by analyzing lines one by one."""
        
        question_lines = []
        options = {}
        correct_answer = "A"
        explanation = ""
        current_section = "question"
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check if this is an option line
            option_match = re.match(r'^([A-Da-d])\s*[\).:\s]\s*(.+)', stripped)
            if option_match:
                current_section = "options"
                letter = option_match.group(1).upper()
                options[letter] = option_match.group(2).strip()
                continue
            
            # Check if this is the correct answer line
            # Handle formats: "Correct Answer: C", "Correct answer: C) text", "Answer: C"
            answer_match = re.match(r'(?:Correct\s*)?[Aa]nswer\s*[.:]\s*([A-Da-d])(?:\s*[\).].*)?$', stripped)
            if answer_match:
                current_section = "answer"
                correct_answer = answer_match.group(1).upper()
                continue
            
            # Check if this is an explanation line
            expl_match = re.match(r'Explanation\s*[.:]\s*(.*)', stripped, re.IGNORECASE)
            if expl_match:
                current_section = "explanation"
                explanation = expl_match.group(1).strip()
                continue
            
            # Otherwise, add to current section
            if current_section == "question":
                # Skip "Question X:" prefix and conversational filler
                cleaned = re.sub(r'^(?:Question\s*\d*[.:]\s*|\d+[.)]\s*)', '', stripped)
                # Skip filler lines
                if cleaned and not re.match(r'(?:Here is|Let me know|Yes,|Now,|I need|I will)', cleaned, re.IGNORECASE):
                    question_lines.append(cleaned)
            elif current_section == "explanation":
                explanation += " " + stripped
        
        question = ' '.join(question_lines).strip()
        
        if question and len(options) == 4:
            return {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": explanation.strip()
            }
        
        return None
    
    def generate_flashcards(self, subject: str, unit: str, count: int = 10, previous_cards: list = None) -> Dict:

        embedding_service = get_embedding_service()
        
        # Get all documents content
        content = embedding_service.get_all_documents_content(subject, unit)
        
        if not content:
            return {
                "status": "error",
                "message": "No content found for this subject/unit"
            }
        
        # Limit content size
        if len(content) > 12000:
            content = content[:12000] + "..."
        
        # Build previous cards context
        previous_context = ""
        if previous_cards and len(previous_cards) > 0:
            previous_context = "\n\nPREVIOUSLY CREATED FLASHCARDS (DO NOT REPEAT OR CREATE SIMILAR TOPICS):\n"
            for i, pc in enumerate(previous_cards, 1):
                previous_context += f"{i}. {pc}\n"
            previous_context += "\nIMPORTANT: Create flashcards on DIFFERENT topics/concepts than the above flashcards.\n"
        
        # Flashcard prompt
        prompt = PromptTemplate(
            input_variables=["content", "count", "previous_context"],
            template="""You are an expert educator creating comprehensive study flashcards. Based on the following educational content, create {count} flashcards that systematically cover the ENTIRE unit.

Content:
{content}
{previous_context}
IMPORTANT INSTRUCTIONS:
1. Cover ALL major topics and subtopics from the content
2. If previous flashcards are provided above, ensure your new flashcards cover COMPLETELY DIFFERENT topics
3. Distribute flashcards across different sections of the material
4. Include a mix of:
   - Key definitions and terminology
   - Important concepts and principles
   - Formulas, equations, or procedures (if applicable)
   - Real-world applications and examples
   - Relationships between concepts
5. Progress from fundamental to advanced concepts
6. Ensure NO duplicate or overlapping content
7. Keep answers concise but informative (2-3 sentences)

For each flashcard, provide:
- Front: A clear question or term
- Back: A concise, accurate answer or definition

Format each flashcard as follows:
Flashcard X:
Front: [question or term]
Back: [concise answer - 2-3 sentences maximum]

---

Create {count} flashcards that comprehensively cover the ENTIRE unit from beginning to end."""
        )
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"content": content, "count": count, "previous_context": previous_context})
            
            # Parse flashcards
            flashcards = self._parse_flashcards(result)
            
            return {
                "status": "success",
                "subject": subject,
                "unit": unit,
                "count": len(flashcards),
                "flashcards": flashcards
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating flashcards: {str(e)}"
            }
    
    def _parse_flashcards(self, flashcard_text: str) -> List[Dict]:

        flashcards = []
        
        # Clean markdown
        cleaned = flashcard_text.replace('**', '').replace('*', '')
        
        # Split by flashcard separator
        cards = re.split(r'\n\s*---+\s*\n', cleaned)
        
        # If only one block, try splitting by "Flashcard X" pattern
        if len(cards) <= 1:
            cards = re.split(r'\n(?=Flashcard\s*\d+)', cleaned)
        
        for card_text in cards:
            if not card_text.strip():
                continue
            
            # Extract front - try multiple patterns
            front = None
            front_patterns = [
                r'[Ff]ront\s*[.:]\s*(.+?)(?=\n\s*[Bb]ack\s*[.:])',
                r'[Qq]uestion\s*[.:]\s*(.+?)(?=\n\s*[Aa]nswer\s*[.:])',
            ]
            for pattern in front_patterns:
                match = re.search(pattern, card_text, re.DOTALL)
                if match:
                    front = match.group(1).strip()
                    break
            
            if not front:
                continue
            
            # Extract back - try multiple patterns
            back = None
            back_patterns = [
                r'[Bb]ack\s*[.:]\s*(.+?)(?=\n\s*---|\n\s*Flashcard|\Z)',
                r'[Aa]nswer\s*[.:]\s*(.+?)(?=\n\s*---|\n\s*Flashcard|\Z)',
            ]
            for pattern in back_patterns:
                match = re.search(pattern, card_text, re.DOTALL)
                if match:
                    back = match.group(1).strip()
                    break
            
            if not back:
                back = ""
            
            if front and back:
                flashcards.append({
                    "front": front,
                    "back": back
                })
        
        return flashcards
    
    def ask_question(self, subject: str, unit: str, question: str) -> Dict:

        embedding_service = get_embedding_service()
        
        # Query relevant documents
        relevant_docs = embedding_service.query_documents(subject, unit, question, n_results=5)
        
        if not relevant_docs:
            return {
                "status": "error",
                "message": "No relevant content found for this question"
            }
        
        # Combine context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        # QA prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful teaching assistant. Answer the student's question based on the provided context.

Context:
{context}

Student's Question: {question}

Instructions:
- Answer clearly and concisely
- Use the context to provide accurate information
- If the context doesn't contain enough information, say "I don't have enough information in the study materials to fully answer this question."
- Be educational and helpful

Answer:"""
        )
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            answer = chain.invoke({"context": context, "question": question})
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "sources": [doc["metadata"].get("source", "Unknown") for doc in relevant_docs]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error answering question: {str(e)}"
            }

# Global instance
_rag_service_instance = None

def get_rag_service() -> RAGService:

    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance

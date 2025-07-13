import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor
from src.nlp import DiseasePredictor
from src.database import fetch_soap_notes, fetch_single_soap_note, get_sqlite_connection
from src.utils import generate_html_response
from src.config import get_config
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors only
logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()
console = Console()

class HIMSCLI:
    """Command-line interface for the HIMS Clinical NLP System."""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='HIMS Clinical NLP System',
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._setup_commands()
    
    def _setup_commands(self):
        subparsers = self.parser.add_subparsers(dest='command')
        status_parser = subparsers.add_parser('status', help='System status')
        status_parser.add_argument('--detail', action='store_true', help='Detailed status')
        
        predict_parser = subparsers.add_parser('predict', help='Run prediction')
        predict_parser.add_argument('text', help='Clinical text to analyze')
        
        process_parser = subparsers.add_parser('process', help='Process SOAP notes')
        process_parser.add_argument('--note-id', type=int, help='Process specific note by ID')
        process_parser.add_argument('--all', action='store_true', help='Process all unprocessed notes')
        process_parser.add_argument('--limit', type=int, default=10, help='Limit number of notes to process')
        process_parser.add_argument('--latest', action='store_true', help='Process the most recently inserted note')
        process_parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
        
        test_parser = subparsers.add_parser('test', help='Run unit tests')
    
    def run(self):
        """Run the CLI with parsed arguments."""
        args = self.parser.parse_args()
        if not args.command:
            self.parser.print_help()
            return
        
        if args.command == 'status':
            self._show_status(args.detail)
        elif args.command == 'predict':
            self._run_prediction(args.text)
        elif args.command == 'process':
            self._process_notes(args.note_id, args.all, args.limit, args.latest, args.parallel)
        elif args.command == 'test':
            self._run_tests()
    
    def _show_status(self, detail: bool = False):
        """Display system status."""
        from src.nlp import DiseasePredictor
        status = {
            "NLP Model": "Loaded" if DiseasePredictor.nlp else "Error",
            "SQLite Database": HIMS_CONFIG["SQLITE_DB_PATH"],
            "UMLS Connection": HIMS_CONFIG["UMLS_DB_URL"],
            "API Endpoint": f"http://{HIMS_CONFIG['API_HOST']}:{HIMS_CONFIG['API_PORT']}",
            "Rate Limit": HIMS_CONFIG["RATE_LIMIT"],
            "Max Workers": HIMS_CONFIG["MAX_WORKERS"]
        }
        
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM soap_notes")
                note_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM soap_notes WHERE ai_analysis IS NOT NULL")
                processed_count = cursor.fetchone()[0]
                status["SOAP Notes"] = f"{processed_count}/{note_count} processed"
                
                cursor.execute("SELECT COUNT(*) FROM diseases")
                status["Diseases"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM symptoms")
                status["Symptoms"] = cursor.fetchone()[0]
        except:
            status["SOAP Notes"] = "Unknown"
        
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        for k, v in status.items():
            table.add_row(k, str(v))
        
        console.print(table)
        
        if detail:
            try:
                with get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, patient_id, created_at,
                               CASE WHEN ai_analysis IS NULL THEN 'Pending' ELSE 'Processed' END AS status
                        FROM soap_notes
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    recent_notes = cursor.fetchall()
                    
                    if recent_notes:
                        note_table = Table(title="Recent SOAP Notes")
                        note_table.add_column("ID", style="cyan")
                        note_table.add_column("Patient ID", style="magenta")
                        note_table.add_column("Created At", style="yellow")
                        note_table.add_column("Status", style="green")
                        
                        for note in recent_notes:
                            note_table.add_row(
                                str(note['id']),
                                note['patient_id'],
                                note['created_at'],
                                note['status']
                            )
                        console.print(note_table)
            except Exception as e:
                console.print(f"[yellow]Couldn't fetch recent notes: {e}[/yellow]")
    
    def _run_prediction(self, text: str):
        """Run disease prediction on input text."""
        from src.nlp import DiseasePredictor
        console.print(Panel("Clinical Text Analysis", style="bold blue"))
        console.print(f"Input: {text[:200]}...\n")
        
        predictor = DiseasePredictor()
        result = predictor.predict_from_text(text)
        
        if not result["primary_diagnosis"] and not result["differential_diagnoses"]:
            console.print("[yellow]No diseases predicted[/yellow]")
            return
        
        if result["primary_diagnosis"]:
            table = Table(title="Primary Diagnosis")
            table.add_column("Disease", style="magenta")
            table.add_column("Score", style="green")
            table.add_row(result["primary_diagnosis"]["disease"], str(result["primary_diagnosis"]["score"]))
            console.print(table)
        
        if result["differential_diagnoses"]:
            table = Table(title="Differential Diagnoses")
            table.add_column("Disease", style="magenta")
            table.add_column("Score", style="green")
            for disease in result["differential_diagnoses"]:
                table.add_row(disease["disease"], str(disease["score"]))
            console.print(table)
    
    def _process_single_note(self, note_id: int) -> bool:
        """Process a single SOAP note."""
        from src.nlp import DiseasePredictor
        from src.database import fetch_single_soap_note, update_ai_analysis
        predictor = DiseasePredictor()
        note = fetch_single_soap_note(note_id)
        if note:
            result = predictor.process_soap_note(note)
            html_content = generate_html_response(result, 200)
            return update_ai_analysis(note["id"], html_content, result['summary'])
        return False
    
    def _process_notes(self, note_id: int, process_all: bool, limit: int, latest: bool = False, parallel: bool = False):
        """Process SOAP notes based on CLI arguments."""
        from src.database import fetch_soap_notes
        if latest:
            console.print(Panel("Processing Latest Note", style="bold green"))
            try:
                with get_sqlite_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM soap_notes ORDER BY created_at DESC LIMIT 1")
                    result = cursor.fetchone()
                    if result:
                        note_id = result['id']
                    else:
                        console.print("[yellow]No notes found in database[/yellow]")
                        return
            except Exception as e:
                logger.error(f"Error fetching latest note: {e}")
                console.print("[red]Failed to fetch latest note[/red]")
                return
        
        if note_id:
            console.print(Panel(f"Processing Note ID: {note_id}", style="bold green"))
            success = self._process_single_note(note_id)
            console.print(f"[green]Successfully processed note {note_id}[/green]" if success else f"[red]Failed to process note {note_id}[/red]")
        elif process_all:
            console.print(Panel("Processing All Notes", style="bold green"))
            notes = fetch_soap_notes()
            if not notes:
                console.print("[yellow]No notes found in database[/yellow]")
                return
                
            notes_to_process = notes[:limit]
            note_ids = [note['id'] for note in notes_to_process]
            
            if parallel:
                console.print(f"[cyan]Using parallel processing with {HIMS_CONFIG['MAX_WORKERS']} workers[/cyan]")
                with ThreadPoolExecutor(max_workers=HIMS_CONFIG["MAX_WORKERS"]) as executor:
                    results = list(track(
                        executor.map(self._process_single_note, note_ids),
                        total=len(note_ids),
                        description="Processing..."
                    ))
                    success_count = sum(results)
            else:
                success_count = 0
                for note in track(notes_to_process, description="Processing..."):
                    try:
                        if self._process_single_note(note['id']):
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to process note {note.get('id')}: {e}")
            
            console.print(f"[green]Successfully processed {success_count}/{len(notes_to_process)} notes[/green]")
        else:
            console.print("[yellow]Specify --note-id, --all, or --latest to process notes[/yellow]")
    
    def _run_tests(self):
        """Run unit tests."""
        from tests.tests import TestNLPApi
        import unittest
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNLPApi)
        unittest.TextTestRunner().run(suite)
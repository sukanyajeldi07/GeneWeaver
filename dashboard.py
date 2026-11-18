
import time

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Button,
    ProgressBar,
    RichLog,
)

from Bio import SeqIO
import pyopencl as cl

from global_alignment_cpu import needleman_wunsch
from global_alignment_gpu import gpu_needleman_wunsch


class GeneWeaverDashboard(App):

    TITLE = "GENEWEAVER - Genomic Control Center"

    CSS = """
    Screen {
        background: #0b1020;
    }

    #heading {
        color: #00e5ff;
        text-style: bold;
        text-align: center;
        padding: 1;
    }

    .panel {
        border: round #315477;
        padding: 1;
        margin: 1;
        height: auto;
    }

    .panel-title {
        color: #00e5ff;
        text-style: bold;
    }

    Button {
        margin: 1;
    }

    #results {
        height: 18;
        border: round #315477;
        padding: 1;
    }

    #status {
        color: #00ff9c;
        padding: 1;
    }

    #progress_text {
        color: #ffffff;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit dashboard")
    ]

    # --------------------------------------------------
    # DASHBOARD LAYOUT
    # --------------------------------------------------

    def compose(self) -> ComposeResult:

        yield Header()

        yield Static(
            "GENEWEAVER | GPU-ASSISTED CRISPR ALIGNMENT ENGINE",
            id="heading"
        )

        # GPU AND ENGINE STATUS

        with Horizontal():

            with Vertical(classes="panel"):

                yield Static(
                    "GPU HARDWARE MONITOR",
                    classes="panel-title"
                )

                yield Static(
                    "Checking OpenCL device...",
                    id="gpu_info"
                )

            with Vertical(classes="panel"):

                yield Static(
                    "ENGINE STATUS",
                    classes="panel-title"
                )

                yield Static(
                    "READY",
                    id="status"
                )

        # ORIGINAL SINGLE DNA INPUT AND PAM DETECTION

        with Container(classes="panel"):

            yield Static(
                "DNA SEQUENCE INPUT & PAM DETECTION",
                classes="panel-title"
            )

            yield Input(
                placeholder="Enter DNA sequence (example: ATGCGTAC)",
                id="dna_input"
            )

            yield Button(
                "Analyze DNA & Detect PAM",
                id="analyze"
            )

        # CPU AND GPU ALIGNMENT

        with Container(classes="panel"):

            yield Static(
                "CPU AND GPU GLOBAL ALIGNMENT",
                classes="panel-title"
            )

            yield Input(
                placeholder="Enter first DNA sequence",
                id="dna1"
            )

            yield Input(
                placeholder="Enter second DNA sequence",
                id="dna2"
            )

            yield Button(
                "Run CPU + GPU Alignment",
                id="align"
            )

        # ORIGINAL FASTA INPUT

        with Container(classes="panel"):

            yield Static(
                "FASTA FILE INPUT",
                classes="panel-title"
            )

            yield Input(
                placeholder="Enter FASTA file path",
                id="fasta_input"
            )

            yield Button(
                "Load FASTA",
                id="load_fasta"
            )

        # PROCESSING PROGRESS

        with Container(classes="panel"):

            yield Static(
                "PROCESSING PROGRESS",
                classes="panel-title"
            )

            yield ProgressBar(
                total=100,
                id="progress"
            )

            yield Static(
                "Waiting for sequence...",
                id="progress_text"
            )

        # RESULTS

        with Container(classes="panel"):

            yield Static(
                "ANALYSIS RESULTS",
                classes="panel-title"
            )

            yield RichLog(
                id="results",
                wrap=True,
                markup=True
            )

        yield Footer()

    # --------------------------------------------------
    # GPU HARDWARE DETECTION
    # --------------------------------------------------

    def on_mount(self):

        self.check_gpu()

    def check_gpu(self):

        info = self.query_one(
            "#gpu_info",
            Static
        )

        try:

            platforms = cl.get_platforms()

            devices = []

            for platform in platforms:

                devices.extend(
                    platform.get_devices(
                        device_type=cl.device_type.GPU
                    )
                )

            if devices:

                device = devices[0]

                memory_mb = (
                    device.global_mem_size
                    / (1024 * 1024)
                )

                info.update(
                    f"[green]GPU DETECTED[/green]\n"
                    f"Device: {device.name}\n"
                    f"Compute Units: {device.max_compute_units}\n"
                    f"Global Memory: {memory_mb:.0f} MB\n"
                    f"Platform: {device.platform.name}"
                )

            else:

                info.update(
                    "[yellow]No OpenCL GPU detected[/yellow]"
                )

        except Exception as error:

            info.update(
                f"GPU information unavailable: {error}"
            )

    # --------------------------------------------------
    # BUTTON EVENTS
    # --------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed):

        if event.button.id == "analyze":

            sequence = self.query_one(
                "#dna_input",
                Input
            ).value

            self.analyze_sequence(sequence)

        elif event.button.id == "align":

            dna1 = self.query_one(
                "#dna1",
                Input
            ).value

            dna2 = self.query_one(
                "#dna2",
                Input
            ).value

            self.run_alignment(dna1, dna2)

        elif event.button.id == "load_fasta":

            path = self.query_one(
                "#fasta_input",
                Input
            ).value

            self.load_fasta(path)

    # --------------------------------------------------
    # ORIGINAL DNA VALIDATION AND PAM DETECTION
    # --------------------------------------------------

    def analyze_sequence(self, sequence):

        sequence = sequence.strip().upper()

        results = self.query_one(
            "#results",
            RichLog
        )

        progress = self.query_one(
            "#progress",
            ProgressBar
        )

        progress_text = self.query_one(
            "#progress_text",
            Static
        )

        status = self.query_one(
            "#status",
            Static
        )

        results.clear()

        progress.update(progress=0)

        if not sequence:

            results.write(
                "[red]Please enter a DNA sequence.[/red]"
            )

            return

        if any(base not in "ATGC" for base in sequence):

            results.write(
                "[red]Invalid DNA sequence. "
                "Use A, T, G, C only.[/red]"
            )

            status.update("VALIDATION FAILED")

            return

        status.update("ANALYZING")

        progress.update(progress=30)

        progress_text.update(
            "DNA validation completed"
        )

        results.write(
            f"[bold cyan]DNA Sequence:[/bold cyan] {sequence}"
        )

        results.write(
            f"Sequence length: {len(sequence)} base pairs"
        )

        progress.update(progress=60)

        # SpCas9 PAM detection: NGG

        pam_sites = [
            i
            for i in range(len(sequence) - 2)
            if sequence[i + 1:i + 3] == "GG"
        ]

        results.write(
            "[bold cyan]SpCas9 PAM Detection (NGG):[/bold cyan]"
        )

        if pam_sites:

            for position in pam_sites:

                results.write(
                    f"[red]PAM found at position {position} "
                    f"(0-based): "
                    f"{sequence[position:position + 3]}[/red]"
                )

        else:

            results.write(
                "No NGG PAM sites found."
            )

        progress.update(progress=100)

        progress_text.update(
            "Analysis completed"
        )

        status.update(
            "ANALYSIS COMPLETE"
        )

    # --------------------------------------------------
    # CPU AND GPU ALIGNMENT COMPARISON
    # --------------------------------------------------

    def run_alignment(self, dna1, dna2):

        dna1 = dna1.strip().upper()
        dna2 = dna2.strip().upper()

        results = self.query_one(
            "#results",
            RichLog
        )

        progress = self.query_one(
            "#progress",
            ProgressBar
        )

        progress_text = self.query_one(
            "#progress_text",
            Static
        )

        status = self.query_one(
            "#status",
            Static
        )

        results.clear()

        progress.update(progress=0)

        if not dna1 or not dna2:

            results.write(
                "[red]Please enter both DNA sequences.[/red]"
            )

            return

        if any(
            base not in "ATGC"
            for base in dna1 + dna2
        ):

            results.write(
                "[red]Invalid DNA sequence. "
                "Use A, T, G, C only.[/red]"
            )

            status.update(
                "VALIDATION FAILED"
            )

            return

        status.update(
            "ALIGNMENT RUNNING"
        )

        try:

            # CPU ALIGNMENT

            progress.update(progress=10)

            progress_text.update(
                "Running CPU alignment..."
            )

            start = time.perf_counter()

            cpu_result = needleman_wunsch(
                dna1,
                dna2
            )

            cpu_time = (
                time.perf_counter() - start
            ) * 1000

            progress.update(progress=40)

            # GPU ALIGNMENT

            progress_text.update(
                "Running Intel OpenCL GPU alignment..."
            )

            start = time.perf_counter()

            gpu_result = gpu_needleman_wunsch(
                dna1,
                dna2
            )

            gpu_time = (
                time.perf_counter() - start
            ) * 1000

            progress.update(progress=80)

            (
                cpu_aligned1,
                cpu_aligned2,
                cpu_score
            ) = cpu_result

            (
                gpu_aligned1,
                gpu_aligned2,
                gpu_score,
                gpu_name
            ) = gpu_result

            # CPU RESULTS

            results.write(
                "[bold cyan]CPU ALIGNMENT RESULTS[/bold cyan]"
            )

            results.write(
                f"Sequence 1: {cpu_aligned1}"
            )

            results.write(
                f"Sequence 2: {cpu_aligned2}"
            )

            results.write(
                f"Alignment Score: {cpu_score}"
            )

            results.write(
                f"CPU Execution Time: {cpu_time:.6f} ms"
            )

            results.write("")

            # GPU RESULTS

            results.write(
                "[bold cyan]GPU ALIGNMENT RESULTS[/bold cyan]"
            )

            results.write(
                f"GPU Device: {gpu_name}"
            )

            results.write(
                f"Sequence 1: {gpu_aligned1}"
            )

            results.write(
                f"Sequence 2: {gpu_aligned2}"
            )

            results.write(
                f"Alignment Score: {gpu_score}"
            )

            results.write(
                f"GPU Execution Time: {gpu_time:.6f} ms"
            )

            results.write("")

            # VERIFY RESULTS

            if (
                cpu_score == gpu_score
                and len(cpu_aligned1) == len(cpu_aligned2)
                and len(gpu_aligned1) == len(gpu_aligned2)
            ):

                results.write(
                    "[green]CPU/GPU alignment scores match.[/green]"
                )

            else:

                results.write(
                    "[red]CPU/GPU result mismatch. "
                    "Further investigation required.[/red]"
                )

            progress.update(progress=100)

            progress_text.update(
                "Alignment comparison completed"
            )

            status.update(
                "ANALYSIS COMPLETE"
            )

        except Exception as error:

            status.update(
                "ALIGNMENT FAILED"
            )

            results.write(
                f"[red]Alignment error: {error}[/red]"
            )

            progress_text.update(
                "Alignment failed"
            )

    # --------------------------------------------------
    # ORIGINAL FASTA FILE PROCESSING
    # --------------------------------------------------

    def load_fasta(self, path):

        results = self.query_one(
            "#results",
            RichLog
        )

        progress = self.query_one(
            "#progress",
            ProgressBar
        )

        progress_text = self.query_one(
            "#progress_text",
            Static
        )

        status = self.query_one(
            "#status",
            Static
        )

        results.clear()

        progress.update(progress=0)

        try:

            records = list(
                SeqIO.parse(path, "fasta")
            )

            if not records:

                results.write(
                    "[red]No FASTA records found.[/red]"
                )

                status.update(
                    "FASTA EMPTY"
                )

                return

            total_records = len(records)

            results.write(
                "[bold cyan]FASTA FILE RESULTS[/bold cyan]"
            )

            for index, record in enumerate(
                records,
                start=1
            ):

                sequence = str(
                    record.seq
                ).upper()

                results.write(
                    f"Record {index}: {record.id}"
                )

                results.write(
                    f"Length: {len(sequence)} base pairs"
                )

                valid = (
                    len(sequence) > 0
                    and all(
                        base in "ATGCN"
                        for base in sequence
                    )
                )

                if valid:

                    results.write(
                        "DNA format: [green]Valid[/green]"
                    )

                else:

                    results.write(
                        "DNA format: [red]Contains unsupported characters[/red]"
                    )

                progress.update(
                    progress=int(
                        index / total_records * 100
                    )
                )

                progress_text.update(
                    f"Processed {index} of "
                    f"{total_records} FASTA records"
                )

            status.update(
                "FASTA PROCESSING COMPLETE"
            )

            progress_text.update(
                "FASTA processing completed"
            )

        except Exception as error:

            status.update(
                "FASTA PROCESSING FAILED"
            )

            results.write(
                f"[red]FASTA error: {error}[/red]"
            )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    GeneWeaverDashboard().run()

"""Shared fixtures for the SoteriaMed test suite."""

import pytest


@pytest.fixture(scope="module")
def sample_chunks():
    """Five StatPearls-shaped chunks, offline, for unit tests.

    The metadata is the real chunk contract -- `chapter_id`,
    `chapter_title`, `section`, `chunk_index` -- not the shape the
    proof-of-concept used. That matters more than it looks: `chunk_index` 0
    recurs across every chapter here, which is precisely the collision that made
    `DecomposingRetriever`'s old merge key silently fold unrelated chunks
    together. Fixtures carrying the shape the code will actually meet are what
    turn that class of bug into a test failure instead of a reading exercise.

    Module-scoped, so the FAISS tests build their index once per module.
    """
    return [
        {
            "text": "A tibial plateau fracture is an intra-articular fracture of "
            "the proximal tibia, usually caused by axial loading. Presenting "
            "features include knee pain, a tense effusion, and inability to bear "
            "weight. Compartment syndrome must be excluded on examination.",
            "metadata": {
                "chapter_id": "SP-0001",
                "chapter_title": "Tibial Plateau Fracture",
                "section": "History and Physical",
                "chunk_index": 0,
            },
        },
        {
            "text": "Substernal chest pain radiating to the left arm or jaw is the "
            "classic presentation of acute coronary syndrome. Diaphoresis, nausea "
            "and dyspnea raise the probability. An ECG showing ST elevation "
            "mandates immediate reperfusion.",
            "metadata": {
                "chapter_id": "SP-0002",
                "chapter_title": "Acute Coronary Syndrome",
                "section": "History and Physical",
                "chunk_index": 0,
            },
        },
        {
            "text": "Colonoscopy is the reference standard for evaluating rectal "
            "bleeding. Adenomatous polyps identified in the sigmoid colon are "
            "removed and sent for histology, since malignant potential rises with "
            "size and degree of dysplasia.",
            "metadata": {
                "chapter_id": "SP-0003",
                "chapter_title": "Colorectal Polyps",
                "section": "Evaluation",
                "chunk_index": 0,
            },
        },
        {
            "text": "Magnetic resonance imaging of the knee demonstrates anterior "
            "cruciate ligament rupture with an associated bone bruise. Arthroscopic "
            "reconstruction is offered to patients returning to pivoting sport.",
            "metadata": {
                "chapter_id": "SP-0004",
                "chapter_title": "Anterior Cruciate Ligament Injury",
                "section": "Evaluation",
                "chunk_index": 0,
            },
        },
        {
            "text": "Reduced ejection fraction on echocardiography, exertional "
            "dyspnea and bilateral peripheral oedema together suggest congestive "
            "heart failure. Diuresis relieves congestion but does not alter "
            "mortality.",
            "metadata": {
                "chapter_id": "SP-0005",
                "chapter_title": "Congestive Heart Failure",
                "section": "History and Physical",
                "chunk_index": 0,
            },
        },
    ]

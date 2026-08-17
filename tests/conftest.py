"""Shared fixtures for the Medical RAG test suite."""

import pytest


@pytest.fixture()
def sample_chunks():
    """Small set of chunks spanning 3 specialties for unit tests."""
    return [
        {
            "text": "Patient presents with acute knee pain after a fall. "
                    "X-ray shows a fracture of the tibial plateau. "
                    "Orthopedic consultation requested for surgical repair.",
            "metadata": {
                "source_index": 0,
                "medical_specialty": "Orthopedic",
                "sample_name": "Knee Fracture",
                "chunk_index": 0,
            },
        },
        {
            "text": "The patient was admitted with substernal chest pain "
                    "radiating to the left arm. ECG shows ST elevation. "
                    "Troponin levels elevated consistent with acute MI.",
            "metadata": {
                "source_index": 1,
                "medical_specialty": "Cardiovascular / Pulmonary",
                "sample_name": "Acute MI",
                "chunk_index": 0,
            },
        },
        {
            "text": "Colonoscopy performed for evaluation of rectal bleeding. "
                    "Multiple polyps found in the sigmoid colon. "
                    "Biopsies taken and sent for pathology.",
            "metadata": {
                "source_index": 2,
                "medical_specialty": "Gastroenterology",
                "sample_name": "Colonoscopy",
                "chunk_index": 0,
            },
        },
        {
            "text": "Right knee MRI reveals a complete tear of the anterior "
                    "cruciate ligament with associated bone bruise. "
                    "Arthroscopic ACL reconstruction recommended.",
            "metadata": {
                "source_index": 3,
                "medical_specialty": "Orthopedic",
                "sample_name": "ACL Tear",
                "chunk_index": 0,
            },
        },
        {
            "text": "Echocardiogram shows ejection fraction of 35 percent. "
                    "Patient has dyspnea on exertion and bilateral lower "
                    "extremity edema consistent with congestive heart failure.",
            "metadata": {
                "source_index": 4,
                "medical_specialty": "Cardiovascular / Pulmonary",
                "sample_name": "CHF",
                "chunk_index": 0,
            },
        },
    ]

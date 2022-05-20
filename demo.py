from easse.cli import report
report(
    'turkcorpus_test',
    sys_sents_path='./easse/resources/data/system_outputs/turkcorpus/test/ACCESS',
    orig_sents_path=None,
    refs_sents_paths=None,
    report_path="easse_report.html",
    tokenizer="13a",
    lowercase=True,
    metrics=['sari', 'sam'],
    sam_type='rc'
)
# Task 1 calendar-suffix deduplication: frozen audit protocol

Date frozen: 5 August 2026

## Status and disclosure

This is a retrospective, post-public-test hypothesis.  The public GRFinNUM
reference answers and the error in public row 70 had already been inspected:
the frozen full-refit prediction contains both `8ης, ΧΡΟΝΙΚΑ` and
`8, ΧΡΟΝΙΚΑ`, whereas the released answer contains only the numeric core.
Consequently, any public-test result from this audit is diagnostic evidence,
not a blind estimate of hidden-test performance.

The proposed rule is nevertheless policy-independent at deployment: it uses
only the input text and predicted entities, never an answer lookup.  Promotion
requires that the released training and validation annotations independently
establish the same annotation convention.

## Frozen inputs

- train parquet SHA-256:
  `8195f2a8430fbe2f5ddd105ae022a2a8af80ef353c28664e2c97d4c91a8a2f24`
- validation parquet SHA-256:
  `1dd14163fb81782e8e82fcb6437509956432cb4d2c8b06bcdf132fccbce19d84`
- released public-test parquet SHA-256:
  `fdfc51bad2bbe84f40a9061ddbb476971120038905c3cf03873839b35b183206`
- frozen validation predictions SHA-256:
  `ce0c474c36a8b7b1430beb9387fb7132fefd38452abf6272728eb54c15c01457`
- frozen public-test predictions SHA-256:
  `2685f95e3d25a9fed8c072d28b6c14337cd913446a00a781c45025207a939aa9`

## Exact hypothesis

In exact Greek calendar syntax of the form
`<one- or two-digit day><η|ης> <Greek month stem>`, GRFinNUM annotates the
numeric core as `ΧΡΟΝΙΚΑ`, not its inflected surface.  The existing
postprocessor already adds that numeric core.  If a token classifier also
emits the inflected surface with the same type, retaining both creates a
redundant false positive.

The change may remove an entity only when all of the following hold:

1. its label is exactly `ΧΡΟΝΙΚΑ`;
2. its surface is exactly the inflected day captured from the input text;
3. that occurrence is immediately followed by a recognized Greek month stem;
4. the same postprocessing pass deterministically emits the captured numeric
   core as `ΧΡΟΝΙΚΑ`.

It must not remove suffix-like predictions outside calendar-month syntax, and
it must not alter entities carrying another label.

## Promotion gates

All gates must pass:

1. Every matching train and validation annotation includes the numeric core as
   `ΧΡΟΝΙΚΑ`; none includes the inflected surface as `ΧΡΟΝΙΚΑ`.
2. At least ten matching train-plus-validation occurrences support the
   convention, and both splits contain support.
3. Applying the rule to the frozen validation predictions changes zero rows
   and leaves validation macro entity F1 exactly unchanged.
4. Unit tests prove removal for both `η` and `ης`, preservation outside month
   syntax, preservation of other labels, and idempotence/no duplication of the
   numeric core.
5. The implementation makes no postal-code, accounting-standard, or other
   unrelated public-error correction.
6. The audit report records public-test use explicitly as `true` and hidden
   competition-test use as `false`.

The released public test is reported only after these convention and
validation gates.  A public improvement alone cannot override a failed gate.


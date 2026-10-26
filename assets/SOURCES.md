# Asset Sources

모든 이미지는 [Unsplash](https://unsplash.com)의 무료 사진이며 [Unsplash License](https://unsplash.com/license)를 따릅니다.
상업·비상업 무료 사용과 수정이 허용되고 출처 표기 의무는 없지만, 보고서·발표 자료에는 아래 출처를 적는 것을 권장합니다.

- 수집일: 2026-10-26
- 배경: 원본을 가로 1600px로 축소 (코드에서 4:3 중앙 크롭 후 640×480으로 변환)
- 물체: 흰 배경 사진을 [rembg](https://github.com/danielgatis/rembg)(`isnet-general-use` 모델)로 배경 제거 → 물체 영역 크롭 → 긴 변 500px 이하로 축소한 투명 PNG

## Backgrounds (`assets/backgrounds/`)

| 파일 | 사진작가 | 원본 |
|---|---|---|
| `bg_desk_01.jpg` | Daniel Liberty | https://unsplash.com/photos/q6A03nMZ4GA |
| `bg_desk_02.jpg` | Tran Mau Tri Tam | https://unsplash.com/photos/4wIlW5Vl82w |
| `bg_desk_03.jpg` | Jess Bailey | https://unsplash.com/photos/xHMmSxmKfIU |
| `bg_desk_04.jpg` | Dragonfly Ave | https://unsplash.com/photos/11s4YFCftKY |
| `bg_craft_01.jpg` | Quang Tran | https://unsplash.com/photos/R3N_pyB8qXo |
| `bg_craft_02.jpg` | Marcus Urbenz | https://unsplash.com/photos/-jW9E2bpeMg |
| `bg_toys_01.jpg` | Vanessa Bucceri | https://unsplash.com/photos/gDiRwIYAMA8 |
| `bg_toys_02.jpg` | Hannah Rodrigo | https://unsplash.com/photos/mf_3yZnC6ug |
| `bg_toys_03.jpg` | Daniel Liberty | https://unsplash.com/photos/rO519nAOPdk |
| `bg_shelf_01.jpg` | Aneta Pawlik | https://unsplash.com/photos/GelF0x5e--c |

## Objects (`assets/objects/`)

| 파일 | 사진작가 | 원본 |
|---|---|---|
| `obj_binderclip_01.png` | Ana | https://unsplash.com/photos/9GmAaEMMKho |
| `obj_coin_01.png` | Collab Media | https://unsplash.com/photos/6kIHGYwvuKI |
| `obj_coin_02.png` | William Warby | https://unsplash.com/photos/8KxrCjUXYcg |
| `obj_dice_01.png` | Joshua Hoehne | https://unsplash.com/photos/Tqwy8Ni4ASY |
| `obj_dice_02.png` | Behnam Norouzi | https://unsplash.com/photos/DOuOfldC6Ao |
| `obj_earbuds_01.png` | Andrey Matveev | https://unsplash.com/photos/clrTR7gDRho |
| `obj_eraser_01.png` | Daniel Shapiro | https://unsplash.com/photos/r5KFjtLvB_I |
| `obj_gluestick_01.png` | charlesdeluvio | https://unsplash.com/photos/AJIJ6Gmy0KM |
| `obj_highlighter_01.png` | andrew davies | https://unsplash.com/photos/IcPus6bjgxE |
| `obj_keys_01.png` | Jozsef Hocza | https://unsplash.com/photos/y5N2HDwagVw |
| `obj_lipstick_01.png` | Giorgio Trovato | https://unsplash.com/photos/ftLmeYwN2p8 |
| `obj_paperclip_01.png` | Ingmar | https://unsplash.com/photos/gkgd6saz8C4 |
| `obj_pencil_01.png` | Kelly Sikkema | https://unsplash.com/photos/g59r724bayQ |
| `obj_rubberduck_01.png` | Timothy Dykes | https://unsplash.com/photos/LhqLdDPcSV8 |
| `obj_spoon_01.png` | Babs Gorniak | https://unsplash.com/photos/065Sxsa3K4c |
| `obj_sunglasses_01.png` | Kiran CK | https://unsplash.com/photos/lSl94SZHRgA |
| `obj_thread_01.png` | Debby Hudson | https://unsplash.com/photos/xFJ8CINzZRQ |
| `obj_toycar_01.png` | Jake | https://unsplash.com/photos/ugkB_S2_O5s |
| `obj_toycar_02.png` | Jake | https://unsplash.com/photos/KTb8UQ-Qd5w |
| `obj_wrench_01.png` | Recha Oktaviani | https://unsplash.com/photos/t__61ap00Mc |

## 파일 이름 규칙

- 배경: `bg_<장면>_<번호>.jpg` (장면: desk, craft, toys, shelf)
- 물체: `obj_<물체>_<번호>.png` (같은 종류가 여러 개면 번호 증가)
- 소문자 영문, 숫자, 밑줄만 사용. 번호는 두 자리(01, 02 …)

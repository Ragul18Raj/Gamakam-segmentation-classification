# Compression robustness -- condition D (cvD), 479 CMD-40 recordings

Generated 2026-10-31 01:16 by `compressing/compress_test.py --summarize`. Each recording is scored by the cvD fold model that never saw it. Ragas: 40. Tonic re-estimated from each compressed file.

Check: stored realcrepe_D .npy top-1 on these recordings 89.4%; the re-extracted original agrees with the stored-input prediction on 98.3% (the stored files carry a different CREPE dither draw).

| level | kbps (median) | x smaller | top-1 % | macro-F1 | top-5 % | same pred as original % | right->wrong | wrong->right | McNemar p | tonic within 20 c % | median pitch agree % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 128 | 1.0 | 89.1 | 0.890 | 93.3 | 100.0 | 0 | 0 | 1 | 100.0 | 100.0 |
| mp3_96k | 96 | 1.3 | 89.1 | 0.890 | 93.7 | 97.9 | 1 | 1 | 1 | 99.4 | 99.5 |
| mp3_56k | 56 | 2.3 | 88.5 | 0.883 | 93.5 | 96.2 | 7 | 4 | 0.549 | 98.7 | 98.2 |
| mp3_32k | 32 | 4.0 | 88.1 | 0.879 | 93.7 | 94.4 | 12 | 7 | 0.359 | 96.7 | 97.4 |
| mp3_24k | 24 | 5.3 | 88.9 | 0.887 | 93.5 | 94.6 | 9 | 8 | 1 | 96.7 | 96.3 |
| mp3_16k | 16 | 8.0 | 87.9 | 0.873 | 93.5 | 91.0 | 20 | 14 | 0.392 | 93.3 | 94.9 |
| mp3_8k | 8 | 16.0 | 77.7 | 0.776 | 88.9 | 78.9 | 68 | 13 | 3.81e-10 | 83.9 | 81.6 |

right->wrong / wrong->right: recordings whose correctness changed relative to the uncompressed file. pitch agree: frames within 50 cents of the uncompressed pitch track. cvD epochs were picked on the test fold, so absolute top-1 is a few points optimistic; the differences between levels are what this test measures.

Recordings lost at mp3_8k:

- bilahari -> sama (tonic +0 c, pitch agree 68%) `Abhishek_Raghuram_December_Season_2010_Karpagambike_Karpagambike`
- shree -> dhanyasi (tonic -501 c, pitch agree 80%) `Amritha_Murali_December_Season_2008_Tyagarajamahadhwajaroha_Tyagarajamahadhwajaroha`
- bhairavi -> karaharapriya (tonic -501 c, pitch agree 85%) `Amritha_Murali_December_Season_2008_Upacharamu_Upacharamu`
- mayamalavagaula -> varali (tonic -711 c, pitch agree 79%) `Amritha_Murali_December_Season_2009_Vidulaku_Vidulaku`
- nata -> sanmukhapriya (tonic -711 c, pitch agree 76%) `Amritha_Murali_December_Season_2010_Pavanatmajagaccha_Pavanatmajagaccha`
- sahana -> kalyani (tonic -711 c, pitch agree 84%) `Amritha_Murali_December_Season_2010_Raghupate_Rama_Raghupate_Rama`
- kapi -> sahana (tonic +0 c, pitch agree 85%) `Amritha_Murali_December_Season_2011_Brindavana_Dolagaaduva_Brindavana_Dolagaaduva`
- sahana -> kalyani (tonic -711 c, pitch agree 84%) `Amrutha_Venkatesh_December_Season_2007_Karunimpa_Idi_Manchi_Karunimpa_Idi_Manchi`
- kanada -> begada (tonic -210 c, pitch agree 79%) `Amrutha_Venkatesh_December_Season_2007_Mamavasada_Janani_Mamavasada_Janani`
- mayamalavagaula -> shree (tonic -711 c, pitch agree 79%) `Amrutha_Venkatesh_December_Season_2007_Vidulaku_Mrokkeda_Vidulaku_Mrokkeda`
- ritigaula -> surati (tonic -711 c, pitch agree 89%) `Amrutha_Venkatesh_December_Season_2008_Janani_Ninnu_Vina_Janani_Ninnu_Vina`
- mukhari -> behag (tonic -882 c, pitch agree 90%) `Amrutha_Venkatesh_December_Season_2009_Entha_Ninne_Entha_Ninne`
- behag -> sindhubhairavi (tonic -811 c, pitch agree 88%) `Amrutha_Venkatesh_December_Season_2010_Narayana_Te_Namo____Narayana_Te_Namo___`
- sencurutti -> sankarabharanam (tonic -701 c, pitch agree 87%) `Aruna_Sairam_December_Season_2002_Pazhani_Shanmuga_Pazhani_Shanmuga`
- bilahari -> sencurutti (tonic +0 c, pitch agree 58%) `D__K__Jayaraman_D_K_Jayaraman_Concert_at_BARC_Mumbai_,_1978_Mamayooram_Meedil_Eriva_Mamayooram_Meedil_Eriva`
- shree -> sindhubhairavi (tonic +10 c, pitch agree 55%) `D__K__Jayaraman_D__K__Jayaraman_concert_at_Shanmukhanantha_Hall,_1985_Srivaralakshmi_Srivaralakshmi`
- sriranjani -> kanada (tonic +0 c, pitch agree 61%) `D__K__Jayaraman_Krishna_Gana_Sabha,_1982_Neesati_Deivamendu_Neesati_Deivamendu`
- sanmukhapriya -> todi (tonic -501 c, pitch agree 92%) `Dr_Pantula_Rama_December_Season_2011_Maamava_Karunaya_Maamava_Karunaya`
- bilahari -> kalyani (tonic -701 c, pitch agree 89%) `Dr_Pantula_Rama_December_Season_2011_Naa_Jeevadhara_Naa_Jeevadhara`
- sankarabharanam -> kalyani (tonic +491 c, pitch agree 85%) `Gayathri_Venkataraghavan_December_Season_2005_Manasu_Swamidhinamayi_Manasu_Swamidhinamayi`
- sama -> behag (tonic -711 c, pitch agree 83%) `Gayathri_Venkataraghavan_December_Season_2006_Varuvaro_Varam_Taruvaro_Varuvaro_Varam_Taruvaro`
- madhyamavati -> sriranjani (tonic -501 c, pitch agree 90%) `Gayathri_Venkataraghavan_December_Season_2010_Adum_Mayilai_-_Karpagame_Adum_Mayilai_-_Karpagame`
- karaharapriya -> madhyamavati (tonic -501 c, pitch agree 88%) `K__V__Narayanaswami_K__V__Narayanaswami_concert_at_Bangalore_1987_Sreenivasa_Thavacharanam_Sreenivasa_Thavacharanam`
- harikambhoji -> sama (tonic +0 c, pitch agree 46%) `K__V__Narayanaswami_K__V__Narayanaswami_concert_at_Dayton,_1993_Dinamanivamsa_Dinamanivamsa`
- atana -> anandabhairavi (tonic +0 c, pitch agree 70%) `K__V__Narayanaswami_Master_Piece_-_Live_Concert_at_Sri_Krishna_Gana_Sabha_Vol_I_Elani_Dayaradu_Elani_Dayaradu`
- sriranjani -> sencurutti (tonic +200 c, pitch agree 87%) `K__V__Narayanaswami_Master_Piece_-_Live_Concert_at_Sri_Krishna_Gana_Sabha_Vol_I_Ini_Oru_Kanam_Ini_Oru_Kanam`
- sencurutti -> saveri (tonic +501 c, pitch agree 84%) `M_R_Subramaniam_December_Season_2008_Kallarkkum_Kallarkkum`
- shree -> dhanyasi (tonic +10 c, pitch agree 72%) `M_S__Subbulakshmi_M_S__Subbulakshmi_concert_at_Naradaganasabha_1969_Endaromahanubhavulu_Endaromahanubhavulu`
- anandabhairavi -> behag (tonic -882 c, pitch agree 92%) `M_S__Subbulakshmi_Paddhatti_Ambikaye_Arul_Purivaai_Ambikaye_Arul_Purivaai`
- sama -> mohanam (tonic -711 c, pitch agree 83%) `Mysore_Manjunath_-_Nagaraj_December_Season_2007_Annapoorne_Visalakshi_Annapoorne_Visalakshi`
- mayamalavagaula -> bhairavi (tonic -701 c, pitch agree 49%) `Nisha_P_Rajagopal_December_Season_2003_Deva_Deva_Kalayami_Deva_Deva_Kalayami`
- harikambhoji -> behag (tonic -711 c, pitch agree 93%) `Nisha_P_Rajagopal_December_Season_2008_Rama_Nannu_Brovaravu_Rama_Nannu_Brovaravu`
- behag -> sindhubhairavi (tonic -801 c, pitch agree 86%) `Nisha_P_Rajagopal_December_Season_2009_Parama_Purusham_Parama_Purusham`
- mukhari -> sindhubhairavi (tonic -501 c, pitch agree 79%) `Nisha_P_Rajagopal_December_Season_2011_Karu_Baru_Karu_Baru`
- purvikalyani -> kamavardani (tonic +0 c, pitch agree 71%) `Nithyasree_Mahadevan_December_Season_2000_Parama_Pavana_Rama_Parama_Pavana_Rama`
- natakurinji -> behag (tonic +481 c, pitch agree 76%) `Nithyasree_Mahadevan_December_Season_2002_Svami_Nan_Unran_Svami_Nan_Unran`
- surati -> ritigaula (tonic -501 c, pitch agree 82%) `Nithyasree_Mahadevan_December_Season_2006_Ento_Premato_Ento_Premato`
- atana -> sriranjani (tonic +701 c, pitch agree 88%) `Nithyasree_Mahadevan_December_Season_2006_Sri_Vaidyanatham_Sri_Vaidyanatham`
- todi -> mayamalavagaula (tonic +0 c, pitch agree 69%) `Nithyasree_Mahadevan_December_Season_2008_Sri_Krishnam_Bhaja_Sri_Krishnam_Bhaja`
- kamavardani -> behag (tonic +0 c, pitch agree 59%) `O__S__Thyagarajan_December_Season_2003_Ninnu_Nera_Nammi_Ninnu_Nera_Nammi`
- kamavardani -> saveri (tonic +0 c, pitch agree 74%) `Prasanna_Venkataraman_December_Season_2009_Vaderadeivamu_Vaderadeivamu`
- husseni -> bhairavi (tonic +0 c, pitch agree 79%) `Ramnad_Krishnan_Ramnad_Krishnan_concert_at_Music_Academy_1969_Enna_Punniyam_Enna_Punniyam`
- sama -> mohanam (tonic -711 c, pitch agree 74%) `S__Sowmya_Carnatic_Vocal_Anna_Purne_Anna_Purne`
- behag -> begada (tonic +0 c, pitch agree 88%) `S__Sowmya_December_Season_1999_Angayarkanni_Angayarkanni`
- nata -> atana (tonic -711 c, pitch agree 86%) `Sangeetha_Swaminathan_December_Season_1999_Jaya_Jaya_Jaya_Janaki_Kantha_Jaya_Jaya_Jaya_Janaki_Kantha`
- mohanam -> sencurutti (tonic +190 c, pitch agree 79%) `Sangeetha_Swaminathan_December_Season_2011_Pahimam_Parvati_Pahimam_Parvati`
- sankarabharanam -> bilahari (tonic -10 c, pitch agree 84%) `Sanjay_Subrahmanyan_Compositions_Of_Muthuswami_Dikshitar_Sundareshwaraya_Namasthe_Sundareshwaraya_Namasthe`
- natakurinji -> sindhubhairavi (tonic -320 c, pitch agree 73%) `Sanjay_Subrahmanyan_December_Season_2000_Chalamela_Chalamela`
- sencurutti -> sindhubhairavi (tonic +170 c, pitch agree 79%) `Sanjay_Subrahmanyan_December_Season_2000_Pittam_Teliya_Pittam_Teliya`
- yadukulakamboji -> sencurutti (tonic +0 c, pitch agree 82%) `Sanjay_Subrahmanyan_December_Season_2008_Kalai_Tukki_Kalai_Tukki`
- mohanam -> begada (tonic +10 c, pitch agree 56%) `Sanjay_Subrahmanyan_Kutcheri,_Live_at_Gokhale_Hall_Kapali_Kapali`
- kalyani -> gaula (tonic +0 c, pitch agree 70%) `Sanjay_Subrahmanyan_Live_at_Vani_Mahal_Paarengum_Paarengum`
- sankarabharanam -> behag (tonic +0 c, pitch agree 75%) `Sanjay_Subrahmanyan_Live_in_USA_-_Sanjay_Subrahmanyan_Shankaracharyam_Shankaracharyam`
- ritigaula -> surati (tonic +511 c, pitch agree 87%) `Sanjay_Subrahmanyan_Sanjay_at_Sastri_Hall_Janani_Janani`
- kalyani -> sencurutti (tonic +701 c, pitch agree 80%) `Sanjay_Subrahmanyan_Sanjay_at_Sastri_Hall_Vanajakshi_-_Varnam_Vanajakshi_-_Varnam`
- karaharapriya -> kedaragaula (tonic +0 c, pitch agree 80%) `Sikkil_Gurucharan_December_Season_2008_Pakkala_Nilabadi_Pakkala_Nilabadi`
- bilahari -> purvikalyani (tonic +0 c, pitch agree 75%) `Sumithra_Vasudev_December_Season_2010_Paridanamiccite_Paridanamiccite`
- sankarabharanam -> kalyani (tonic -701 c, pitch agree 87%) `Sumithra_Vasudev_December_Season_2011_Sankaracharyam_Sankaracharyam`
- kamavardani -> mayamalavagaula (tonic -10 c, pitch agree 79%) `Sumithra_Vasudev_Tasmai_Sri_Gurave_Namaha_Sri_Gurupadanakha_Sri_Gurupadanakha`
- yadukulakamboji -> sankarabharanam (tonic +0 c, pitch agree 89%) `T__Brinda_All_Time_Favourites_-_Vol_1_Ninnusevinchina_Ninnusevinchina`
- mukhari -> bhairavi (tonic +0 c, pitch agree 86%) `T__Brinda_All_Time_Favourites_-_Vol_2_Emanine_Emanine`
- dhanyasi -> todi (tonic +0 c, pitch agree 79%) `T__Brinda_All_Time_Favourites_-_Vol_2_Talachina_Talachina`
- todi -> dhanyasi (tonic +0 c, pitch agree 80%) `T__M__Krishna_December_Season_1999_Vazhi_Maraittirukkude_Vazhi_Maraittirukkude`
- purvikalyani -> begada (tonic +0 c, pitch agree 70%) `T__M__Krishna_December_Season_2006_Ninni_Vinaga_Ninni_Vinaga`
- madhyamavati -> sindhubhairavi (tonic +10 c, pitch agree 72%) `T__M__Krishna_Madrasil_Margazhi_2006_Alakalella_Alakalella`
- sriranjani -> sencurutti (tonic -671 c, pitch agree 57%) `T__M__Krishna_Vaishnava_Ksetra_Karumugil_Vannam_-Badrinath_Karumugil_Vannam_-Badrinath`
- ritigaula -> madhyamavati (tonic -701 c, pitch agree 87%) `Vasundara_Rajagopal_Vasundara_Rajagopal_at_Arkay_Ninnuvina_Marigalada_Ninnuvina_Marigalada`
- dhanyasi -> purvikalyani (tonic +0 c, pitch agree 74%) `Vijay_Siva_Madrasil_Margazhi_2006_Rama_Dayajudave_Rama_Dayajudave`

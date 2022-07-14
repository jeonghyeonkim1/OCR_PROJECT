from PIL import Image
from streamlit_drawable_canvas import st_canvas
from matplotlib import font_manager, rc
from modules.methods import *
from functools import reduce
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit.components.v1 as components
import numpy as np
import streamlit as st
import seaborn as sns
import os

sns.set(rc={'figure.figsize': (15, 8)}, font_scale=2)

@st.cache(allow_output_mutation=True)
def load_model():
    character = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝'
    saved_model = 'C:/models/kor_model/best_loss_0.04_model.h5'
    model = VGG_FeatureExtractor(len(character))
    model.load_weights(saved_model)

    return model, list(character)


model, idx2char = load_model()

if 'typing' not in st.session_state:
    st.session_state.typing = ''

if 'skip' not in st.session_state:
    st.session_state.skip = '0'

if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = ''

if 'select_shop' not in st.session_state:
    st.session_state.select_shop = '0'

if st.session_state.ocr_result == "":
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    if bg_image == None:
        st.stop()
        

    img = np.array(Image.open(bg_image))

    st.image(img)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    bookstore_list = {'kyobo': '교보문고', 'yp': '영풍문고', 'arcnbook': '아크앤북스', 'jongnobooks': '종로서적'}

    preds = [bookstore_list[i] for i in all_process(img) if i in bookstore_list]
    
    with st.form("my_form"):
        options = st.multiselect(
            '원하시는 매장을 선택해주세요!',
            preds,
            preds
        )

        submit_btn = st.form_submit_button("검색")

        if not submit_btn:
            st.stop()
        elif submit_btn and len(options) > 1:
            st.write(" #### 한 가지만 선택해주세요!!")
            st.stop()
        elif submit_btn and len(options) == 0:
            st.write(" #### 한 가지 선택해주세요!!")
            st.stop()
        elif submit_btn and len(options) == 1:
            st.session_state.ocr_result = options[0]
            st.experimental_rerun()

else:
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider(
            "Point display radius: ", 1, 25, 8)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#FFC0CB")
    realtime_update = st.sidebar.checkbox("Update in realtime", False)

    st.info(f"검색한 서점 : {st.session_state.ocr_result}")

    shop_list = reduce(lambda a, b: a.append(b) or a if b.find(st.session_state.typing) != -1 else a, ['광화문점', '가든파이브 바로드림센터', '강남점', '건대 바로드림센터', '동대문 바로드림센터',
        '디큐브시티 바로드림센터', '목동점', '서울대 교내서점', '수유 바로드림센터', '신논현역스토어',
        '영등포점', '은평 바로드림센터', '이화여대 교내서점', '잠실점', '천호점', '청량리 바로드림센터',
        '합정점', '광교점', '광교 월드스퀘어센터', '부천점', '분당점', '송도 바로드림센터', '인천점',
        '일산점', '판교점', '평촌점', '경성대ㆍ부경대 센터', '광주상무 센터', '대구점', '대전점', '부산점',
        '세종 바로드림센터', '센텀시티점', '울산점', '전북대 교내서점', '전주 바로드림센터', '창원점', 
        '천안점', '칠곡 센터', '포항공대 교내서점', '해운대센터'], [])

    with st.form("my_form"):
        area = st.multiselect(
            '원하시는 매장을 선택해주세요!',
            shop_list,
            shop_list
        )

        area_btn = st.form_submit_button("검색")

        if area_btn:
            st.session_state.select_shop = '1'

            shop_info = [
                {
                    "shop_name": i.text.strip(),
                    "shop_num": re.sub('[^\d]', '', i.attrs['href'])
                }
                for i in BeautifulSoup(requests.get("http://www.kyobobook.co.kr/storen/MainStore.laf?SITE=01&Kc=GNHHNOoffstore&orderClick=rvd").text, 'html.parser').select('.store_list li a')
                if i.text.strip() in area
            ]

            book_info = [
                {
                    "book_name": i.attrs['alt'],
                    "book_img": i.attrs['src']
                }
                for i in BeautifulSoup(requests.get(f"http://mkiosk.kyobobook.co.kr/kiosk/index.ink?site=01&mode=pc").text, 'html.parser').select('.today_book .rolling img')
            ]


            for book in book_info:
                st.image(book['book_img'])
                st.write(f" #### {book['book_name']}")
                
                with st.expander("책 재고 보기"):
                    for shop in shop_info:
                        st.write(f" #### <{shop['shop_name']}>")
                        st.write(f"{find_book_num(book['book_name'], shop['shop_num'])}")
                

    if st.session_state.select_shop == '0':
        col1, col2 = st.columns(2)
        btn1, btn2, btn3, btn4, btn5 = st.columns(5)

        with col1:
            canvas = st_canvas(
                fill_color='#FFFFFF',
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                update_streamlit=realtime_update,
                width=288,
                height=288,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                display_toolbar=st.sidebar.checkbox("Display toolbar", True),
                key='canvas'
            )

        with btn2:
            if st.button("띄어쓰기"):
                st.session_state.typing += " "
                st.session_state.skip = "1"

        with btn3:
            if st.button("지움"):
                st.session_state.typing = st.session_state.typing[:-1]
                st.session_state.skip = "1"

        with btn4:
            if st.button("초기화"):
                st.session_state.typing = ""
                st.session_state.skip = "1"

        if canvas.image_data is not None:

            # image load
            canvas = canvas.image_data.astype('uint8')
            canvas = cv2.resize(canvas, (32, 32))

            # preprocess image
            x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            x = np.array(x, dtype=np.float32)
            x = x.reshape((-1, 32, 32, 3))
            x = x / 255.

            # predict
            y = model.predict(x).squeeze()
            result = tf.argmax(y)

            if st.session_state.skip == "1":
                st.session_state.skip = "0"
            else:
                st.session_state.typing += idx2char[result] if idx2char[result] != '랬' and idx2char[result] != '웝' else ""

            with col2:
                st.write(f'<div style="height: 288px; font-size: 30px; background: pink; padding: 10px;">{st.session_state.typing}</div>', unsafe_allow_html=True)

            # show result
            st.write(f' ## Result: {idx2char[result]}' if idx2char[result]
                        != '랬' and idx2char[result] != '웝' else ' ## 글자를 적어주세요!')

            # show prediction of most five
            most_arg = y.argsort()[::-1][:5]
            most_val = [f'{y[idx]*100:.8f}' for idx in most_arg]
            chars = [f'{idx2char[idx]}' for idx in most_arg]

            chart_data = pd.DataFrame(
                np.array([most_val, chars]).T,
                columns=['Prob(%)', 'Pred']
            )

            font_path = "c:/Windows/Fonts/malgun.ttf"
            font = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font)
            fig, ax = plt.subplots()
            ax.bar(
                chart_data['Pred'],
                chart_data['Prob(%)'].apply(lambda a: round(float(a), 1)),
                color='red',
                alpha=0.5
            )
            st.pyplot(fig)



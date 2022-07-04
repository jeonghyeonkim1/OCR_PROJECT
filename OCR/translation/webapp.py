from PIL import Image
from streamlit_drawable_canvas import st_canvas
from matplotlib import font_manager, rc
from modules.methods import *
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import os
import time
import string
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

DB_HOST = "localhost"
DB_USER = "myuser118"
DB_PASSWORD = "1234"
DB_NAME = "mydb118"

# st.snow()
sns.set(rc = {'figure.figsize':(15,8)}, font_scale = 2)
result_texts = []

if 'typing' not in st.session_state:
	st.session_state.typing = ''

if 'skip' not in st.session_state:
	st.session_state.skip = '0'

if 'success' not in st.session_state:
	st.session_state.success = '0'

if 'question' not in st.session_state:
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

    sql = '''
        SELECT sent FROM rand_sent ORDER BY RAND()
    '''

    with db.cursor() as cursor:
        cursor.execute(sql)
        st.session_state.question = cursor.fetchone()[0]
    
    db.close()


@st.cache(allow_output_mutation=True)
def load_model():
    character = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝'
    saved_model = 'C:/models/kor_model/best_loss_0.04_model.h5'
    model = VGG_FeatureExtractor(len(character))
    model.load_weights(saved_model)
    
    return model, list(character)

model, idx2char = load_model()

if st.session_state.success == '1':
    st.success('성공하셨습니다!! 다음 문제를 푸시겠습니까?')
    if st.button("넹!!!!"):
        db = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            passwd=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8'
        )

        sql = f'''
            SELECT sent FROM rand_sent WHERE sent != '{st.session_state.question} ORDER BY RAND()'
        '''

        with db.cursor() as cursor:
            cursor.execute(sql)
            st.session_state.question = cursor.fetchone()[0]
        
        db.close()

        st.session_state.success = '0'
        st.session_state.typing = ''

    if st.session_state.success != '0':
        st.stop()

quest = st.empty()

if len(st.session_state.typing) == len(st.session_state.question):
    st.exception(RuntimeError("틀렸습니다."))

col1, col2 = st.columns(2)
btn1, btn2, btn3, btn4, btn5 = st.columns(5)

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 8)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#FFC0CB")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", False)

if bg_image != None:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # image info -------------------------------------------------
    img = np.array(Image.open(bg_image))
    height, width, channel = img.shape

    # contour maker -------------------------------------------------
    weight_path = 'C:/models/contour_maker'
    weight_name = os.listdir(weight_path)[0]
    weight_loc = os.path.join(weight_path, weight_name)

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default=weight_loc,
                        type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7,
                        type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float,
                        help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4,
                        type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool,
                        help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280,
                        type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float,
                        help='image magnification ratio')
    parser.add_argument('--poly', default=False,
                        action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False,
                        action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='data/',
                        type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False,
                        action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth',
                        type=str, help='pretrained refiner model')

    args = parser.parse_args()

    net = CRAFT()
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(
            torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if args.refine:
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(
            	copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(
            	torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    image = loadImage(img)

    bboxes, polys = test_net(
        args,
        net,
        image,
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        args.cuda,
        args.poly,
        refine_net
    )

    saveResult(image[:, :, ::-1], polys)

    print("elapsed time : {}s".format(time.time() - t))

    # text-recognizer -------------------------------------------------
    model_path = 'C:/models/text_recognizer'
    model_name = os.listdir(model_path)[0]
    model_loc = os.path.join(model_path, model_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='result/',
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--saved_model', default=model_loc,
                        help="path to saved_model to evaluation")
    # Data processing
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    # Model Architecture
    parser.add_argument('--Transformation', type=str,
                        default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    def demo(opt):
        """ model configuration """
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(
            imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor(
                    [opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(
                    batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)

                else:
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                return [re.sub(r'\[s\]', '', pred) for pred in preds_str]

    preds = demo(opt)

    st.image(img)

    with st.form("my_form"):
        options = st.multiselect(
            '선택하신 매장의 메뉴를 검색해드립니다!',
            preds,
            preds
        )

        # Every form must have a submit button.
        submitted = st.form_submit_button("검색")
        if submitted:
            for i in options:
                st.write(f"선택하신 매장: {i}")


else:
    quest.info(f'다음 문장을 입력하십시오 : {st.session_state.question}')

    with col1:
        st.write(' ')
        st.write(' ')
        canvas = st_canvas(
            fill_color='#FFFFFF',
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
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
        img = canvas.image_data.astype('uint8')
        img = cv2.resize(img, (32, 32))
        
        # preprocess image
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            message = st.text_area(
                label = '',
                value = st.session_state.typing,
                height = 288,
                disabled = True
            )

        if st.session_state.typing == st.session_state.question:
            st.session_state.success = '1'
        else:
            st.session_state.success = '0'

        # show result
        st.write(f' ## Result: {idx2char[result]}' if idx2char[result] != '랬' and idx2char[result] != '웝' else ' ## 글자를 적어주세요!')

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

            
                    
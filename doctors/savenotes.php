<?php
session_start(); 
include('../connect.php');

$a = date('Y-m-d H:i:s');
$j = $_POST['pn'];
$b = $_POST['notes'];
$doctor =$_SESSION['SESS_FIRST_NAME'];
$sql = "INSERT INTO patient_notes (created_at,patient,notes,posted_by) VALUES (:a,:b,:c,:d)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$j,':c'=>$b,':d'=>$doctor));

$reset=0;
$sql = "UPDATE patients
SET  served=?
WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($reset,$j)); 

?>
<?php
if (isset($_POST['dx'])) {
$dxs = $_POST['dx'];
foreach ($dxs as $dx) {
$sql = "INSERT INTO dx (patient, disease) VALUES(:a,:b)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $j,
':b' => $dx
));
}
}
if (isset($_POST['lab'])) {
$labs = $_POST['lab'];
foreach ($labs as $lab) {
$sql = "INSERT INTO lab (test,opn,reqby) VALUES (:a,:b,:c)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $lab,
':b' => $j,
':c' => $doctor
));
}
}
 
    if (isset($_GET['islab'])) {
        $reset=1;
$sql = "UPDATE patients
SET  served=?
WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($reset,$j));

//update lab to remove patient from the doctors waiting list
$reset=0;
$sql = "UPDATE lab
SET  served=?
WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($reset,$j));
}
$reset=0;
$sql = "UPDATE patients
SET  served=?
WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($reset,$j));

if (isset($_POST['image'])) {
$images = $_POST['image'];
foreach ($images as $image) {
$sql = "INSERT INTO req_images (test,opn,reqby) VALUES (:a,:b,:c)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $image,
':b' => $j,
':c' => $doctor
));
}
}
 $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$j));
$code=mt_rand(10000000, 99999999);
header("location: prescribe_inp.php?search=$j&response=0&code=$code");

?>



